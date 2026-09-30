use parking_lot::Mutex;
use thiserror::Error;

use crate::{
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        kernel::{
            LogitTransformKernel,
            matmul::{MatmulA, MatmulArguments, MatmulDOps, MatmulKernel},
        },
    },
    config::{embedding::AnyEmbeddingConfig, weight_matrix::AnyWeightMatrixSpec},
    data_type::DataType,
    encodable_block::{
        embedding_table::{EmbeddingTable, EmbeddingTableError, read_output_signs},
        linear::{Gather, LinearMatmulError, UntiedReadout},
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum EmbeddingError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Unsupported configuration: {0}")]
    UnsupportedConfiguration(String),
    #[error("Embedding table error: {0}")]
    EmbeddingTable(#[from] EmbeddingTableError<B>),
    #[error(transparent)]
    LinearMatmul(#[from] LinearMatmulError<B>),
}

/// Tied readout multiplies by the lookup table's own matrix; untied readout owns a separate one.
enum Readout<B: Backend> {
    Tied(Mutex<<B::Kernels as Kernels>::MatmulKernel>),
    Untied(UntiedReadout<B>),
}

pub struct Embedding<B: Backend> {
    table: EmbeddingTable<B>,
    readout: Readout<B>,
    input_scale: f32,
    data_type: DataType,
    logit_transform: Option<LogitTransform<B>>,
    vocab_size: u32,
    model_dim: u32,
}

struct LogitTransform<B: Backend> {
    scale: f32,
    soft_cap: Option<f32>,
    kernel: <B::Kernels as Kernels>::LogitTransformKernel,
}

impl<B: Backend> Embedding<B> {
    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    pub fn model_dim(&self) -> u32 {
        self.model_dim
    }

    pub fn new(
        context: &B::Context,
        vocab_size: u32,
        model_dim: u32,
        config: &AnyEmbeddingConfig,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<(Self, Option<Allocation<B>>), EmbeddingError<B>> {
        let (table, readout, readout_input_hadamard_factors) = match config {
            AnyEmbeddingConfig::TiedEmbeddingConfig(_) => {
                let embedding_tree = parameter_tree.subtree("embedding");
                let table = EmbeddingTable::load(context, &embedding_tree, vocab_size, model_dim, data_type)?;
                if table.as_matrix().is_none() {
                    return Err(EmbeddingError::UnsupportedConfiguration("tied embeddings need a matrix table".into()));
                }
                // The output norm fuses the readout's input rotation, so it needs its own copy of the factors.
                let readout_input_hadamard_factors =
                    if let AnyWeightMatrixSpec::HybridSpec(_) = embedding_tree.metadata("spec")? {
                        Some(read_output_signs(&embedding_tree, model_dim)?)
                    } else {
                        None
                    };
                let kernel = <B::Kernels as Kernels>::MatmulKernel::new(context, data_type, data_type, data_type)
                    .map_err(EmbeddingError::BackendError)?;
                (table, Readout::Tied(Mutex::new(kernel)), readout_input_hadamard_factors)
            },
            AnyEmbeddingConfig::UntiedEmbeddingConfig(_) => {
                let input_embedding_tree = parameter_tree.subtree("input_embedding");
                let table = EmbeddingTable::load(context, &input_embedding_tree, vocab_size, model_dim, data_type)?;
                let output_embedding_tree = parameter_tree.subtree("output_embedding");
                let output_embedding_spec = output_embedding_tree.metadata("spec")?;
                let output = UntiedReadout::load(
                    context,
                    &output_embedding_tree,
                    output_embedding_spec,
                    vocab_size,
                    model_dim,
                    data_type,
                )?;
                (table, Readout::Untied(output), None)
            },
        };

        let input_scale = config.input_scale().unwrap_or(1.0);
        let logit_scale = config.logit_scale().unwrap_or(1.0);
        let logit_soft_cap = *config.logit_soft_cap();
        let logit_transform = if logit_scale != 1.0 || logit_soft_cap.is_some() {
            let kernel =
                <B::Kernels as Kernels>::LogitTransformKernel::new(context, data_type, logit_soft_cap.is_some())
                    .map_err(EmbeddingError::BackendError)?;
            Some(LogitTransform {
                scale: logit_scale,
                soft_cap: logit_soft_cap,
                kernel,
            })
        } else {
            None
        };

        Ok((
            Self {
                table,
                readout,
                input_scale,
                data_type,
                logit_transform,
                vocab_size,
                model_dim,
            },
            readout_input_hadamard_factors,
        ))
    }

    pub fn encode_lookup(
        &self,
        token_ids: &Allocation<B>,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, EmbeddingError<B>> {
        encoder.push_debug_group("embedding lookup");

        let mut output = encoder
            .allocate_scratch_for_shape(&[batch_dim, self.model_dim], self.data_type)
            .map_err(EmbeddingError::BackendError)?;

        self.table.encode_lookup(token_ids, &mut output, batch_dim, self.input_scale, encoder);

        encoder.pop_debug_group();

        Ok(output)
    }

    pub fn encode_readout(
        &self,
        batch_dim: u32,
        input_allocation: &Allocation<B>,
        output_dim: u32,
        gather_indices: Option<&Allocation<B>>,
        apply_logit_transform: bool,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, EmbeddingError<B>> {
        encoder.push_debug_group("embedding readout");

        assert!(batch_dim > 0 && output_dim > 0, "Embedding readout requires non-empty dimensions");
        let mut output_allocation = match &self.readout {
            Readout::Untied(output) => {
                let gather = gather_indices.map(|indices| Gather {
                    indices,
                    output_dim,
                });
                output.encode(input_allocation, batch_dim, gather, encoder).map_err(EmbeddingError::BackendError)?
            },
            Readout::Tied(readout) => {
                let matrix = self.table.as_matrix().expect("tied embedding tables are matrices");
                let mut output = encoder
                    .allocate_scratch_for_shape(&[batch_dim, output_dim], self.data_type)
                    .map_err(EmbeddingError::BackendError)?;
                let arguments = MatmulArguments {
                    a: MatmulA::FullPrecision {
                        values: input_allocation,
                        offset: 0,
                    },
                    b: matrix.matmul_b(),
                    b_leading_dimension: None,
                    b_transpose: true,
                    d: &mut output,
                    d_transform: MatmulDOps::none(),
                    gather_indices,
                    m: batch_dim,
                    n: output_dim,
                    k: self.model_dim,
                };
                readout.lock().encode(arguments, encoder).map_err(EmbeddingError::BackendError)?;
                output
            },
        };

        if apply_logit_transform && let Some(logit_transform) = &self.logit_transform {
            let length = batch_dim * output_dim;
            logit_transform.kernel.encode(
                &mut output_allocation,
                length,
                logit_transform.scale,
                logit_transform.soft_cap.unwrap_or(0.0),
                encoder,
            );
        }

        encoder.pop_debug_group();

        Ok(output_allocation)
    }
}
