use thiserror::Error;

use crate::{
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        kernel::{FullPrecisionEmbeddingLookupKernel, MiraiSEmbeddingLookupKernel, QuantizedEmbeddingLookupKernel},
    },
    config::weight_matrix::{AnyWeightMatrixSpec, Layout},
    data_type::DataType,
    encodable_block::weight_matrix::{WeightMatrix, WeightMatrixError},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum EmbeddingTableError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Weight matrix error: {0}")]
    WeightMatrix(#[from] WeightMatrixError<B>),
    #[error("Unsupported embedding table configuration: {0}")]
    UnsupportedConfiguration(String),
}

enum Lookup<B: Backend> {
    FullPrecision {
        kernel: <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel,
        matrix: WeightMatrix<B>,
    },
    Quantized {
        kernel: <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel,
        matrix: WeightMatrix<B>,
        output_hadamard_factors: Option<Allocation<B>>,
    },
    /// Lookup-only D4 lattice table (`D4S4Spec`).
    MiraiS {
        kernel: <B::Kernels as Kernels>::MiraiSEmbeddingLookupKernel,
        codes: Allocation<B>,
        row_scales: Allocation<B>,
        ladder_indices: Allocation<B>,
        ladder: Allocation<B>,
        table: Allocation<B>,
        output_hadamard_factors: Allocation<B>,
    },
}

pub struct EmbeddingTable<B: Backend> {
    lookup: Lookup<B>,
    vocab_size: u32,
    embedding_dim: u32,
}

impl<B: Backend> EmbeddingTable<B> {
    pub fn load(
        context: &B::Context,
        tree: &ParameterTree<B>,
        vocab_size: u32,
        embedding_dim: u32,
        data_type: DataType,
    ) -> Result<Self, EmbeddingTableError<B>> {
        let spec = tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        Self::load_with_spec(context, tree, vocab_size, embedding_dim, data_type, spec, None)
    }

    pub fn load_with_spec(
        context: &B::Context,
        tree: &ParameterTree<B>,
        vocab_size: u32,
        embedding_dim: u32,
        data_type: DataType,
        spec: AnyWeightMatrixSpec,
        output_hadamard_factors: Option<Allocation<B>>,
    ) -> Result<Self, EmbeddingTableError<B>> {
        if let AnyWeightMatrixSpec::D4S4Spec(spec) = spec {
            assert!(output_hadamard_factors.is_none());
            if spec.layout != Layout::InputOutput || data_type != DataType::BF16 || !embedding_dim.is_multiple_of(128) {
                return Err(EmbeddingTableError::UnsupportedConfiguration(format!(
                    "{spec:?} with {data_type:?} and embedding dim {embedding_dim}"
                )));
            }
            let read = |name: &str, shape: &[u32], data_type: DataType| {
                tree.leaf(name)?.validate(shape, data_type)?.read_allocation()
            };
            let lookup = Lookup::MiraiS {
                kernel: <B::Kernels as Kernels>::MiraiSEmbeddingLookupKernel::new(context)
                    .map_err(EmbeddingTableError::BackendError)?,
                codes: read("codes", &[vocab_size, embedding_dim / 4], DataType::U8)?,
                row_scales: read("row_scales", &[vocab_size], DataType::BF16)?,
                ladder_indices: read("ladder_indices", &[vocab_size, embedding_dim / 128], DataType::U8)?,
                ladder: read("ladder", &[16], DataType::F16)?,
                table: read("table", &[256, 4], DataType::I8)?,
                output_hadamard_factors: read("output_hadamard_factors", &[embedding_dim], DataType::I32)?,
            };
            return Ok(Self {
                lookup,
                vocab_size,
                embedding_dim,
            });
        }

        let matrix = WeightMatrix::load(tree, spec, Layout::InputOutput, embedding_dim, vocab_size, data_type)?;
        if output_hadamard_factors.is_some() && matrix.quantization().is_none() {
            return Err(EmbeddingTableError::UnsupportedConfiguration(
                "output-hadamard factors require a quantized table".into(),
            ));
        }

        let lookup = match matrix.quantization() {
            None => Lookup::FullPrecision {
                kernel: <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel::new(context, data_type)
                    .map_err(EmbeddingTableError::BackendError)?,
                matrix,
            },
            Some(info) => Lookup::Quantized {
                kernel: <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel::new(
                    context,
                    data_type,
                    info.group_size,
                    info.mode,
                    info.method,
                    output_hadamard_factors.is_some(),
                )
                .map_err(EmbeddingTableError::BackendError)?,
                matrix,
                output_hadamard_factors,
            },
        };

        Ok(Self {
            lookup,
            vocab_size,
            embedding_dim,
        })
    }

    pub fn matrix(&self) -> &WeightMatrix<B> {
        match &self.lookup {
            Lookup::FullPrecision {
                matrix,
                ..
            }
            | Lookup::Quantized {
                matrix,
                ..
            } => matrix,
            Lookup::MiraiS {
                ..
            } => panic!("Mirai S embedding tables are lookup-only"),
        }
    }

    /// Gathers one row per token id into `output`, scaling by `scale`.
    pub fn encode_lookup(
        &self,
        token_ids: &Allocation<B>,
        output: &mut Allocation<B>,
        batch_dim: u32,
        scale: f32,
        encoder: &mut Encoder<B>,
    ) {
        match &self.lookup {
            Lookup::FullPrecision {
                kernel,
                matrix,
            } => kernel.encode(
                token_ids,
                matrix.values(),
                output,
                batch_dim,
                self.vocab_size,
                self.embedding_dim,
                scale,
                encoder,
            ),
            Lookup::Quantized {
                kernel,
                matrix,
                output_hadamard_factors,
            } => kernel.encode(
                token_ids,
                matrix.values(),
                matrix.scales().expect("quantized lookup requires scales"),
                matrix.zero_points(),
                matrix.biases(),
                output,
                output_hadamard_factors.as_ref(),
                batch_dim,
                self.vocab_size,
                self.embedding_dim,
                scale,
                encoder,
            ),
            Lookup::MiraiS {
                kernel,
                codes,
                row_scales,
                ladder_indices,
                ladder,
                table,
                output_hadamard_factors,
            } => kernel.encode(
                token_ids,
                codes,
                row_scales,
                ladder_indices,
                ladder,
                table,
                output_hadamard_factors,
                output,
                batch_dim,
                self.vocab_size,
                self.embedding_dim,
                scale,
                encoder,
            ),
        }
    }
}
