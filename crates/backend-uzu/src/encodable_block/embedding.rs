use parking_lot::Mutex;
use thiserror::Error;

use crate::{
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::HADAMARD_TRANSFORM_BLOCK_SIZE,
        kernel::{
            ActivationTransform, LogitSoftCapKernel,
            matmul::{MatmulA, MatmulArguments, MatmulDOps, MatmulKernel},
        },
    },
    config::{
        embedding::AnyEmbeddingConfig,
        weight_matrix::{
            AnyWeightMatrixSpec, Layout,
            hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
        },
    },
    data_type::DataType,
    encodable_block::{
        embedding_table::{EmbeddingTable, EmbeddingTableError},
        weight_matrix::{WeightMatrix, WeightMatrixError},
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
    #[error("Weight matrix error: {0}")]
    WeightMatrix(#[from] WeightMatrixError<B>),
}

struct UntiedReadout<B: Backend> {
    matrix: WeightMatrix<B>,
    readout: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
    input_hadamard: Option<InputHadamard<B>>,
}

struct InputHadamard<B: Backend> {
    factors: Allocation<B>,
    kernel: ActivationTransform<B>,
}

enum EmbeddingTying<B: Backend> {
    Tied {
        table: EmbeddingTable<B>,
        readout: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
    },
    Untied {
        input_table: EmbeddingTable<B>,
        output: UntiedReadout<B>,
    },
}

pub struct Embedding<B: Backend> {
    tying: EmbeddingTying<B>,
    input_scale: f32,
    data_type: DataType,
    logit_soft_cap: Option<LogitSoftCap<B>>,
    vocab_size: u32,
    model_dim: u32,
}

struct LogitSoftCap<B: Backend> {
    value: f32,
    kernel: <B::Kernels as Kernels>::LogitSoftCapKernel,
}

impl<B: Backend> Embedding<B> {
    pub(crate) fn data_type(&self) -> DataType {
        self.data_type
    }

    pub(crate) fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    pub(crate) fn model_dim(&self) -> u32 {
        self.model_dim
    }

    fn readout_input_hadamard(&self) -> Option<&InputHadamard<B>> {
        match &self.tying {
            EmbeddingTying::Untied {
                output,
                ..
            } => output.input_hadamard.as_ref(),
            EmbeddingTying::Tied {
                ..
            } => None,
        }
    }

    fn readout_operands(&self) -> (&WeightMatrix<B>, &Mutex<<B::Kernels as Kernels>::MatmulKernel>) {
        match &self.tying {
            EmbeddingTying::Tied {
                table,
                readout,
            } => (table.matrix(), readout),
            EmbeddingTying::Untied {
                output,
                ..
            } => (&output.matrix, &output.readout),
        }
    }

    pub fn new(
        context: &B::Context,
        vocab_size: u32,
        model_dim: u32,
        config: &AnyEmbeddingConfig,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<(Self, Option<Allocation<B>>), EmbeddingError<B>> {
        let (tying, readout_input_hadamard_factors) = match config {
            AnyEmbeddingConfig::TiedEmbeddingConfig(_) => {
                let embedding_tree = parameter_tree.subtree("embedding")?;
                let embedding_spec = embedding_tree.metadata::<AnyWeightMatrixSpec>("spec")?;

                let (tying, readout_input_hadamard_factors) = match embedding_spec {
                    spec @ (AnyWeightMatrixSpec::FullPrecisionSpec(_)
                    | AnyWeightMatrixSpec::MLXSpec(_)
                    | AnyWeightMatrixSpec::IntSpec(_)) => {
                        let table = EmbeddingTable::load_with_spec(
                            context,
                            &embedding_tree,
                            vocab_size,
                            model_dim,
                            data_type,
                            spec,
                            None,
                        )?;

                        (
                            EmbeddingTying::Tied {
                                table,
                                readout: readout_kernel(context, data_type)?,
                            },
                            None,
                        )
                    },
                    AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                        quantization_spec,
                        adapter_spec: None,
                        incoherence_block_size: Some(HADAMARD_TRANSFORM_BLOCK_SIZE),
                        incoherence_processing_mode: IncoherenceProcessingMode::Output,
                        ..
                    }) => {
                        let incoherence_signs_tree = embedding_tree.subtree("incoherence_signs")?;
                        let output_hadamard_factors = Some(
                            incoherence_signs_tree
                                .leaf("output_signs")?
                                .validate(&[model_dim], DataType::I32)?
                                .read_allocation()?,
                        );
                        let readout_input_hadamard_factors = Some(
                            incoherence_signs_tree
                                .leaf("output_signs")?
                                .validate(&[model_dim], DataType::I32)?
                                .read_allocation()?,
                        );

                        let table = EmbeddingTable::load_with_spec(
                            context,
                            &embedding_tree.subtree("quantized")?,
                            vocab_size,
                            model_dim,
                            data_type,
                            *quantization_spec,
                            output_hadamard_factors,
                        )?;
                        (
                            EmbeddingTying::Tied {
                                table,
                                readout: readout_kernel(context, data_type)?,
                            },
                            readout_input_hadamard_factors,
                        )
                    },
                    spec => return Err(EmbeddingError::UnsupportedConfiguration(format!("{spec:?}"))),
                };

                (tying, readout_input_hadamard_factors)
            },
            AnyEmbeddingConfig::UntiedEmbeddingConfig(_) => {
                let input_embedding_tree = parameter_tree.subtree("input_embedding")?;
                let input_embedding_spec = input_embedding_tree.metadata::<AnyWeightMatrixSpec>("spec")?;

                let input_table = match input_embedding_spec {
                    AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                        quantization_spec,
                        adapter_spec: None,
                        incoherence_block_size: Some(HADAMARD_TRANSFORM_BLOCK_SIZE),
                        incoherence_processing_mode: IncoherenceProcessingMode::Output,
                        ..
                    }) => {
                        let output_hadamard_factors = Some(
                            input_embedding_tree
                                .subtree("incoherence_signs")?
                                .leaf("output_signs")?
                                .validate(&[model_dim], DataType::I32)?
                                .read_allocation()?,
                        );
                        EmbeddingTable::load_with_spec(
                            context,
                            &input_embedding_tree.subtree("quantized")?,
                            vocab_size,
                            model_dim,
                            data_type,
                            *quantization_spec,
                            output_hadamard_factors,
                        )?
                    },
                    spec => EmbeddingTable::load_with_spec(
                        context,
                        &input_embedding_tree,
                        vocab_size,
                        model_dim,
                        data_type,
                        spec,
                        None,
                    )?,
                };

                let output_embedding_tree = parameter_tree.subtree("output_embedding")?;
                let output_embedding_spec = output_embedding_tree.metadata::<AnyWeightMatrixSpec>("spec")?;

                let output = match output_embedding_spec {
                    AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                        quantization_spec,
                        adapter_spec: None,
                        incoherence_block_size: Some(HADAMARD_TRANSFORM_BLOCK_SIZE),
                        incoherence_processing_mode: IncoherenceProcessingMode::Input,
                        ..
                    }) => {
                        let matrix = WeightMatrix::load(
                            &output_embedding_tree.subtree("quantized")?,
                            *quantization_spec,
                            Layout::OutputInput,
                            vocab_size,
                            model_dim,
                            data_type,
                        )?;

                        // Input-side incoherence is applied privately to the readout
                        // input: the shared hidden state must stay untransformed
                        // (e.g. for the speculator).
                        let factors = output_embedding_tree
                            .subtree("incoherence_signs")?
                            .leaf("input_signs")?
                            .validate(&[model_dim], DataType::I32)?
                            .read_allocation()?;
                        let kernel = ActivationTransform::input_rht(context, data_type, false)
                            .map_err(EmbeddingError::BackendError)?;

                        UntiedReadout {
                            matrix,
                            readout: readout_kernel(context, data_type)?,
                            input_hadamard: Some(InputHadamard {
                                factors,
                                kernel,
                            }),
                        }
                    },
                    spec => {
                        let matrix = WeightMatrix::load(
                            &output_embedding_tree,
                            spec,
                            Layout::OutputInput,
                            vocab_size,
                            model_dim,
                            data_type,
                        )?;
                        UntiedReadout {
                            matrix,
                            readout: readout_kernel(context, data_type)?,
                            input_hadamard: None,
                        }
                    },
                };

                (
                    EmbeddingTying::Untied {
                        input_table,
                        output,
                    },
                    None,
                )
            },
        };

        let input_scale = config.input_scale().unwrap_or(1.0);
        let logit_soft_cap = if let Some(value) = *config.logit_soft_cap() {
            let kernel = <B::Kernels as Kernels>::LogitSoftCapKernel::new(context, data_type)
                .map_err(EmbeddingError::BackendError)?;
            Some(LogitSoftCap {
                value,
                kernel,
            })
        } else {
            None
        };

        Ok((
            Self {
                tying,
                input_scale,
                data_type,
                logit_soft_cap,
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
            .allocate_scratch_with_shape(&[batch_dim, self.model_dim], self.data_type)
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
        table.encode_lookup(token_ids, &mut output, batch_dim, self.input_scale, encoder);

        encoder.pop_debug_group();

        Ok(output)
    }

    pub fn encode_readout(
        &self,
        batch_dim: u32,
        input_allocation: &Allocation<B>,
        output_data_type: DataType,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, EmbeddingError<B>> {
        encoder.push_debug_group("embedding readout");

        assert!(batch_dim > 0, "Embedding readout requires at least one row");
        let native_output = output_data_type == self.data_type;
        let input_hadamard = self.readout_input_hadamard();
        let mut output_allocation = encoder
            .allocate_scratch_with_shape(&[batch_dim, self.vocab_size], output_data_type)
            .map_err(EmbeddingError::BackendError)?;

        let (matrix, readout) = self.readout_operands();
        let mut rht_input: Option<Allocation<B>> = None;
        let a = match input_hadamard {
            Some(input_hadamard) => {
                let mut transformed =
                    encoder.allocate_scratch(input_allocation.size()).map_err(EmbeddingError::BackendError)?;
                input_hadamard.kernel.encode_fp(
                    input_allocation,
                    &mut transformed,
                    &input_hadamard.factors,
                    batch_dim,
                    self.model_dim,
                    encoder,
                );
                rht_input.insert(transformed)
            },
            None => input_allocation,
        };
        let arguments = MatmulArguments {
            a: MatmulA::FullPrecision {
                values: a,
                offset: 0,
            },
            b: matrix.matmul_b(),
            b_leading_dimension: None,
            b_transpose: true,
            d: &mut output_allocation,
            d_transform: MatmulDOps::none(),
            gather_indices: None,
            m: batch_dim,
            n: self.vocab_size,
            k: self.model_dim,
        };
        if native_output {
            readout.lock().encode(arguments, encoder).map_err(EmbeddingError::BackendError)?;
        } else {
            let mut widened = <B::Kernels as Kernels>::MatmulKernel::new(
                encoder.context(),
                self.data_type,
                self.data_type,
                output_data_type,
            )
            .map_err(EmbeddingError::BackendError)?;
            widened.encode(arguments, encoder).map_err(EmbeddingError::BackendError)?;
        }

        if let Some(logit_soft_cap) = &self.logit_soft_cap {
            let length = batch_dim * self.vocab_size;
            if native_output {
                logit_soft_cap.kernel.encode(&mut output_allocation, length, logit_soft_cap.value, encoder);
            } else {
                let kernel = <B::Kernels as Kernels>::LogitSoftCapKernel::new(encoder.context(), output_data_type)
                    .map_err(EmbeddingError::BackendError)?;
                kernel.encode(&mut output_allocation, length, logit_soft_cap.value, encoder);
            }
        }

        encoder.pop_debug_group();

        Ok(output_allocation)
    }

    /// Per-row candidate readout via the GEMV B-row gather: `out[r][j] == dense[r][token_ids[r][j]]`,
    /// soft-capped when configured, one dispatch. Caller guarantees `token_ids < vocab_size`.
    pub(crate) fn encode_readout_sparse(
        &self,
        input: &Allocation<B>,
        token_ids: &Allocation<B>,
        rows: usize,
        ids_per_row: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, EmbeddingError<B>> {
        encoder.push_debug_group("embedding readout (sparse)");

        assert!(rows > 0 && ids_per_row > 0);
        let input_hadamard = self.readout_input_hadamard();
        let (matrix, readout) = self.readout_operands();
        let b = matrix.matmul_b();

        let mut output = encoder
            .allocate_scratch_with_shape(&[rows as u32, ids_per_row as u32], self.data_type)
            .map_err(EmbeddingError::BackendError)?;

        let mut rht_input: Option<Allocation<B>> = None;
        let a = match input_hadamard {
            Some(input_hadamard) => {
                let mut transformed = encoder.allocate_scratch(input.size()).map_err(EmbeddingError::BackendError)?;
                input_hadamard.kernel.encode_fp(
                    input,
                    &mut transformed,
                    &input_hadamard.factors,
                    rows as u32,
                    self.model_dim,
                    encoder,
                );
                rht_input.insert(transformed)
            },
            None => input,
        };

        let soft_cap = self.logit_soft_cap.as_ref().map(|cap| cap.value);
        readout
            .lock()
            .encode(
                MatmulArguments {
                    a: MatmulA::FullPrecision {
                        values: a,
                        offset: 0,
                    },
                    b,
                    b_leading_dimension: None,
                    b_transpose: true,
                    d: &mut output,
                    d_transform: MatmulDOps {
                        soft_cap,
                        ..MatmulDOps::none()
                    },
                    gather_indices: Some(token_ids),
                    m: rows as u32,
                    n: ids_per_row as u32,
                    k: self.model_dim,
                },
                encoder,
            )
            .map_err(EmbeddingError::BackendError)?;

        encoder.pop_debug_group();

        Ok(output)
    }
}

fn readout_kernel<B: Backend>(
    context: &B::Context,
    data_type: DataType,
) -> Result<Mutex<<B::Kernels as Kernels>::MatmulKernel>, EmbeddingError<B>> {
    let kernel = <B::Kernels as Kernels>::MatmulKernel::new(context, data_type, data_type, data_type)
        .map_err(EmbeddingError::BackendError)?;
    Ok(Mutex::new(kernel))
}
