use thiserror::Error;

use crate::{
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        kernel::{FullPrecisionEmbeddingLookupKernel, QuantizedEmbeddingLookupKernel},
    },
    config::weight_matrix::{AnyWeightMatrixSpec, Layout},
    data_type::DataType,
    encodable_block::weight_matrix::{QuantizationInfo, WeightMatrix, WeightMatrixError},
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

enum LookupKernel<B: Backend> {
    FullPrecision(<B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel),
    Quantized(<B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel),
}

pub struct EmbeddingTable<B: Backend> {
    matrix: WeightMatrix<B>,
    lookup: LookupKernel<B>,
    output_hadamard_factors: Option<Allocation<B>>,
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
        let matrix = WeightMatrix::load(tree, spec, Layout::InputOutput, embedding_dim, vocab_size, data_type)?;
        if output_hadamard_factors.is_some() && matrix.quantization().is_none() {
            return Err(EmbeddingTableError::UnsupportedConfiguration(
                "output-hadamard factors require a quantized table".into(),
            ));
        }

        let lookup = match matrix.quantization() {
            None => {
                let kernel = <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel::new(context, data_type)
                    .map_err(EmbeddingTableError::BackendError)?;
                LookupKernel::FullPrecision(kernel)
            },
            Some(QuantizationInfo::Microfloat(_)) => {
                return Err(EmbeddingTableError::UnsupportedConfiguration(
                    "microfloat embedding tables are not supported".into(),
                ));
            },
            Some(QuantizationInfo::Integer {
                mode,
                method,
                group_size,
            }) => {
                let kernel = <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel::new(
                    context,
                    data_type,
                    group_size,
                    mode,
                    method,
                    output_hadamard_factors.is_some(),
                )
                .map_err(EmbeddingTableError::BackendError)?;
                LookupKernel::Quantized(kernel)
            },
        };

        Ok(Self {
            matrix,
            lookup,
            output_hadamard_factors,
            vocab_size,
            embedding_dim,
        })
    }

    pub fn matrix(&self) -> &WeightMatrix<B> {
        &self.matrix
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
            LookupKernel::FullPrecision(kernel) => kernel.encode(
                token_ids,
                self.matrix.values(),
                output,
                batch_dim,
                self.vocab_size,
                self.embedding_dim,
                scale,
                encoder,
            ),
            LookupKernel::Quantized(kernel) => kernel.encode(
                token_ids,
                self.matrix.values(),
                self.matrix.scales().expect("quantized lookup requires scales"),
                self.matrix.zero_points(),
                self.matrix.biases(),
                output,
                self.output_hadamard_factors.as_ref(),
                batch_dim,
                self.vocab_size,
                self.embedding_dim,
                scale,
                encoder,
            ),
        }
    }
}
