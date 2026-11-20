use std::sync::Arc;

use derive_more::Debug;
use parking_lot::Mutex;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
        kernel::{
            LogitTransformKernel,
            matmul::{MatmulA, MatmulArguments, MatmulDOps, MatmulGather, MatmulKernel, MatmulOutput},
        },
    },
    config::{
        embedding::AnyEmbeddingConfig,
        weight_matrix::{
            AnyWeightMatrixSpec,
            hybrid_spec::{AnyIncoherenceProcessorSpec, IncoherenceProcessingMode},
        },
    },
    data_type::DataType,
    encodable_block::{
        embedding_table::{EmbeddingTable, EmbeddingTableError},
        linear::{
            InputAccess, InputPreparationMode, LinearBlockError, LinearEncodeError, OutputBasis, OutputRows,
            OutputStack,
        },
        weight_matrix::WeightMatrix,
    },
    safetensors::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum EmbeddingError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Matmul error: {0}")]
    Matmul(#[source] <<B::Kernels as Kernels>::MatmulKernel as MatmulKernel>::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Embedding table error: {0}")]
    EmbeddingTable(#[from] EmbeddingTableError<B>),
    #[error(transparent)]
    LinearBlock(#[from] LinearBlockError<B>),
    #[error("Linear encoding error: {0}")]
    LinearEncode(#[from] LinearEncodeError<B>),
}

enum EmbeddingTying<B: Backend> {
    Tied {
        table: EmbeddingTable<B>,
        matrix: Arc<WeightMatrix<B>>,
        readout: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
    },
    Untied {
        input_table: EmbeddingTable<B>,
        output: OutputStack<B>,
    },
}

pub struct Embedding<B: Backend> {
    tying: EmbeddingTying<B>,
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

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn readout_input_hadamard_factors(&self) -> Option<Arc<B::GlobalGpuBuffer>> {
        match &self.tying {
            EmbeddingTying::Tied {
                table,
                ..
            } => table.readout_input_hadamard_factors(),
            EmbeddingTying::Untied {
                ..
            } => None,
        }
    }

    pub async fn new(
        vocab_size: u32,
        model_dim: u32,
        data_type: DataType,
        embedding_config: &AnyEmbeddingConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<(Self, Option<Arc<B::GlobalGpuBuffer>>), EmbeddingError<B>> {
        if *embedding_config.permute_indices() {
            parameter_tree.ignore("index_permutation", &[vocab_size], DataType::I32)?;
        }

        let tying = match embedding_config {
            AnyEmbeddingConfig::TiedEmbeddingConfig(_) => {
                let embedding_tree = parameter_tree.subtree("embedding");
                let (table, matrix) =
                    EmbeddingTable::load_tied(vocab_size, model_dim, data_type, &embedding_tree, context).await?;
                EmbeddingTying::Tied {
                    table,
                    matrix,
                    readout: Mutex::new(
                        <B::Kernels as Kernels>::MatmulKernel::new(context, data_type, data_type, data_type)
                            .map_err(EmbeddingError::Matmul)?,
                    ),
                }
            },
            AnyEmbeddingConfig::UntiedEmbeddingConfig(_) => {
                let input_embedding_tree = parameter_tree.subtree("input_embedding");
                let input_table =
                    EmbeddingTable::load(vocab_size, model_dim, data_type, &input_embedding_tree, context).await?;
                let output_embedding_tree = parameter_tree.subtree("output_embedding");
                let output_embedding_spec = output_embedding_tree.metadata("spec")?;
                if let AnyWeightMatrixSpec::HybridSpec(hybrid) = &output_embedding_spec {
                    let unsupported = match &hybrid.incoherence_spec {
                        Some(AnyIncoherenceProcessorSpec::HadamardSpec(p))
                            if p.processing_mode == IncoherenceProcessingMode::InputOutput =>
                        {
                            Some("readout supports input-only Hadamard transforms")
                        },
                        _ => None,
                    };
                    if let Some(message) = unsupported {
                        return Err(LinearBlockError::UnsupportedConfiguration(message.into()).into());
                    }
                }
                let (output, _) = OutputStack::load(
                    model_dim,
                    vocab_size,
                    data_type,
                    data_type,
                    data_type,
                    output_embedding_spec,
                    &output_embedding_tree,
                    None,
                    InputPreparationMode::InLinear,
                    InputAccess::Preserve,
                    OutputBasis::Original,
                    context,
                )
                .await?;
                EmbeddingTying::Untied {
                    input_table,
                    output,
                }
            },
        };

        let input_scale = embedding_config.input_scale().unwrap_or(1.0);
        let logit_scale = embedding_config.logit_scale().unwrap_or(1.0);
        let logit_soft_cap = *embedding_config.logit_soft_cap();
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

        let embedding = Self {
            tying,
            input_scale,
            data_type,
            logit_transform,
            vocab_size,
            model_dim,
        };
        let signs = embedding.readout_input_hadamard_factors();
        Ok((embedding, signs))
    }

    pub fn encode_lookup(
        &self,
        token_ids: impl BufferRef<Backend = B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, EmbeddingError<B>> {
        let mut output = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, self.model_dim], self.data_type)
            .map_err(EmbeddingError::BackendError)?;

        let table = match &self.tying {
            EmbeddingTying::Tied {
                table,
                ..
            } => table,
            EmbeddingTying::Untied {
                input_table,
                ..
            } => input_table,
        };
        table.encode_lookup(token_ids, &mut output, batch_dim, self.input_scale, command_buffer);

        Ok(output)
    }

    pub fn encode_readout(
        &self,
        input_buffer: impl BufferRef<Backend = B>,
        batch_dim: u32,
        output_dim: u32,
        gather_indices: Option<impl BufferRef<Backend = B>>,
        apply_logit_transform: bool,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, EmbeddingError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("embedding readout");

        assert!(
            batch_dim > 0 && output_dim > 0 && (gather_indices.is_some() || output_dim <= self.vocab_size),
            "invalid embedding readout dimensions"
        );
        let mut output_buffer = match &self.tying {
            EmbeddingTying::Untied {
                output,
                ..
            } => {
                let rows = match gather_indices {
                    Some(indices) => OutputRows::Gather {
                        indices,
                        count: output_dim,
                    },
                    None => OutputRows::Prefix(output_dim),
                };
                output.encode_input(MatmulA::full_precision(input_buffer), batch_dim, rows, command_buffer)?
            },
            EmbeddingTying::Tied {
                matrix,
                readout,
                ..
            } => {
                let mut output = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim, output_dim], self.data_type)
                    .map_err(EmbeddingError::BackendError)?;
                let arguments = MatmulArguments {
                    a: MatmulA::FullPrecision {
                        values: input_buffer,
                        offset: 0,
                    },
                    b: matrix.matmul_b(),
                    b_leading_dimension: None,
                    b_transpose: true,
                    output: MatmulOutput::new(&mut output, MatmulDOps::none()),
                    gather: gather_indices.map(|indices| MatmulGather {
                        indices,
                        first_row: 0,
                        row_count: self.vocab_size,
                    }),
                    m: batch_dim,
                    n: output_dim,
                    k: self.model_dim,
                };
                readout.lock().encode(arguments, command_buffer).map_err(EmbeddingError::Matmul)?;
                output
            },
        };

        if apply_logit_transform && let Some(logit_transform) = &self.logit_transform {
            let length = batch_dim * output_dim;
            logit_transform.kernel.encode(
                &mut output_buffer,
                length,
                logit_transform.scale,
                logit_transform.soft_cap.unwrap_or(0.0),
                command_buffer,
            );
        }

        Ok(output_buffer)
    }
}
