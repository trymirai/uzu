use half::{bf16, f16};
use parking_lot::Mutex;
use thiserror::Error;

use crate::{
    backends::common::{
        Allocation, AllocationType, Backend, Context, Encoder, Kernels,
        gpu_types::{HADAMARD_TRANSFORM_BLOCK_SIZE, QuantizationMode},
        kernel::{
            ActivationTransform, LogitTransformKernel,
            matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            qtip_s_exact::{I3S4ReadoutArguments, I3S4SparseReadoutArguments, QtipSExactKernel},
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

/// i3_s4 head repacked at load time into the standard symmetric group-64
/// integer GEMM format (4-bit or 8-bit signed codes, bf16 group scales
/// = bf16(row_scale * ladder[index])). Runs the generic Uzu GEMM instead of
/// the staged i3 readout kernel.
struct I3Gemm<B: Backend> {
    codes: Allocation<B>,
    scales: Allocation<B>,
    mode: QuantizationMode,
    readout: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
    rht: ActivationTransform<B>,
    factors: Allocation<B>,
}

struct UntiedReadout<B: Backend> {
    i3_gemm: Option<I3Gemm<B>>,
    matrix: Option<WeightMatrix<B>>,
    readout: Option<Mutex<<B::Kernels as Kernels>::MatmulKernel>>,
    input_hadamard: Option<InputHadamard<B>>,
    i3_s4: Option<I3S4Readout<B>>,
}

struct I3S4Readout<B: Backend> {
    kernel: <B::Kernels as Kernels>::QtipSExactKernel,
    codes: Allocation<B>,
    row_scales: Allocation<B>,
    ladder_indices: Allocation<B>,
    ladder: Allocation<B>,
    input_hadamard_factors: Allocation<B>,
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
    logit_transform: Option<LogitTransform<B>>,
    vocab_size: u32,
    model_dim: u32,
}

struct LogitTransform<B: Backend> {
    scale: f32,
    soft_cap: Option<f32>,
    kernel: <B::Kernels as Kernels>::LogitTransformKernel,
    widened_kernel: Option<<B::Kernels as Kernels>::LogitTransformKernel>,
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
            } => (
                output.matrix.as_ref().expect("i3 readout has no WeightMatrix"),
                output.readout.as_ref().expect("i3 readout has no generic matmul"),
            ),
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
                let embedding_tree = parameter_tree.subtree("embedding");
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
                        incoherence_block_size: Some(block_size),
                        incoherence_processing_mode: IncoherenceProcessingMode::Output,
                        ..
                    }) if block_size == HADAMARD_TRANSFORM_BLOCK_SIZE => {
                        let incoherence_signs_tree = embedding_tree.subtree("incoherence_signs");
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
                            &embedding_tree.subtree("quantized"),
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
                let input_embedding_tree = parameter_tree.subtree("input_embedding");
                let input_embedding_spec = input_embedding_tree.metadata::<AnyWeightMatrixSpec>("spec")?;

                let input_table = match input_embedding_spec {
                    AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                        quantization_spec,
                        adapter_spec: None,
                        incoherence_block_size: Some(block_size),
                        incoherence_processing_mode: IncoherenceProcessingMode::Output,
                        ..
                    }) if block_size == HADAMARD_TRANSFORM_BLOCK_SIZE => {
                        let output_hadamard_factors = Some(
                            input_embedding_tree
                                .subtree("incoherence_signs")
                                .leaf("output_signs")?
                                .validate(&[model_dim], DataType::I32)?
                                .read_allocation()?,
                        );
                        EmbeddingTable::load_with_spec(
                            context,
                            &input_embedding_tree.subtree("quantized"),
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

                let output_embedding_tree = parameter_tree.subtree("output_embedding");
                let output_embedding_spec = output_embedding_tree.metadata::<AnyWeightMatrixSpec>("spec")?;

                let output = match output_embedding_spec {
                    AnyWeightMatrixSpec::I3S4Spec(spec) => {
                        assert_eq!(spec.layout, Layout::OutputInput);
                        let head_mode = std::env::var("QTIP_I3_HEAD").unwrap_or_else(|_| "u4".to_string());
                        // tiered head (INT4 hot rows / INT3 / INT2 cold rows) announces itself with `band_bounds`
                        let band_bounds = output_embedding_tree
                            .leaf("band_bounds")
                            .ok()
                            .map(|leaf| leaf.validate(&[2], DataType::U32).and_then(|leaf| leaf.read_slice::<u32>()))
                            .transpose()?;
                        assert!(
                            band_bounds.is_none() || matches!(head_mode.as_str(), "u4" | "u8"),
                            "tiered i3 head needs QTIP_I3_HEAD=u4|u8"
                        );
                        let i3_gemm = match head_mode.as_str() {
                            "u4" | "u8" => {
                                let row_scales = output_embedding_tree
                                    .leaf("row_scales")?
                                    .validate(&[vocab_size], DataType::BF16)?
                                    .read_slice::<bf16>()?;
                                let ladder_indices = output_embedding_tree
                                    .leaf("ladder_indices")?
                                    .validate(&[vocab_size, model_dim / 128], DataType::U8)?
                                    .read_slice::<u8>()?;
                                let ladder = output_embedding_tree
                                    .leaf("ladder")?
                                    .validate(&[16], DataType::F16)?
                                    .read_slice::<f16>()?;
                                let eight_bit = head_mode == "u8";
                                let (codes, scales) = if let Some(bounds) = &band_bounds {
                                    let (hot, cold) = (bounds[0], bounds[1]);
                                    let codes4 = output_embedding_tree
                                        .leaf("codes4")?
                                        .validate(&[hot, model_dim / 2], DataType::U8)?
                                        .read_slice::<u8>()?;
                                    let codes3 = output_embedding_tree
                                        .leaf("codes")?
                                        .validate(&[cold - hot, model_dim * 3 / 8], DataType::U8)?
                                        .read_slice::<u8>()?;
                                    let codes2: Vec<u8> = if cold < vocab_size {
                                        output_embedding_tree
                                            .leaf("codes2")?
                                            .validate(&[vocab_size - cold, model_dim / 4], DataType::U8)?
                                            .read_slice::<u8>()?
                                            .to_vec()
                                    } else {
                                        Vec::new()
                                    };
                                    eprintln!("tiered head: INT4 rows < {hot}, INT3 rows < {cold}, INT2 rows >= {cold}");
                                    repack_tiered_to_symmetric_gemm(
                                        hot as usize,
                                        cold as usize,
                                        &codes4,
                                        &codes3,
                                        &codes2,
                                        &row_scales,
                                        &ladder_indices,
                                        &ladder,
                                        vocab_size as usize,
                                        model_dim as usize,
                                        eight_bit,
                                    )
                                } else {
                                    let codes3 = output_embedding_tree
                                        .leaf("codes")?
                                        .validate(&[vocab_size, model_dim * 3 / 8], DataType::U8)?
                                        .read_slice::<u8>()?;
                                    repack_i3_s4_to_symmetric_gemm(
                                        &codes3,
                                        &row_scales,
                                        &ladder_indices,
                                        &ladder,
                                        vocab_size as usize,
                                        model_dim as usize,
                                        eight_bit,
                                    )
                                };
                                let mut codes_allocation = context
                                    .create_allocation(codes.len(), AllocationType::Global)
                                    .map_err(EmbeddingError::BackendError)?;
                                codes_allocation.copyin(&codes);
                                let mut scales_allocation = context
                                    .create_allocation(scales.len() * 2, AllocationType::Global)
                                    .map_err(EmbeddingError::BackendError)?;
                                scales_allocation.copyin(&scales);
                                Some(I3Gemm {
                                    codes: codes_allocation,
                                    scales: scales_allocation,
                                    mode: if eight_bit {
                                        QuantizationMode::U8
                                    } else {
                                        QuantizationMode::U4
                                    },
                                    readout: readout_kernel(context, data_type)?,
                                    rht: ActivationTransform::input_rht(context, data_type, false)
                                        .map_err(EmbeddingError::BackendError)?,
                                    factors: output_embedding_tree
                                        .leaf("input_hadamard_factors")?
                                        .validate(&[model_dim], DataType::I32)?
                                        .read_allocation()?,
                                })
                            },
                            _ => None,
                        };
                        let tiered = band_bounds.is_some();
                        UntiedReadout {
                            i3_gemm,
                            matrix: None,
                            readout: None,
                            input_hadamard: None,
                            i3_s4: if tiered { None } else { Some(I3S4Readout {
                                kernel: <B::Kernels as Kernels>::QtipSExactKernel::new(context)
                                    .map_err(EmbeddingError::BackendError)?,
                                codes: output_embedding_tree
                                    .leaf("codes")?
                                    .validate(&[vocab_size, model_dim * 3 / 8], DataType::U8)?
                                    .read_allocation()?,
                                row_scales: output_embedding_tree
                                    .leaf("row_scales")?
                                    .validate(&[vocab_size], DataType::BF16)?
                                    .read_allocation()?,
                                ladder_indices: output_embedding_tree
                                    .leaf("ladder_indices")?
                                    .validate(&[vocab_size, model_dim / 128], DataType::U8)?
                                    .read_allocation()?,
                                ladder: output_embedding_tree
                                    .leaf("ladder")?
                                    .validate(&[16], DataType::F16)?
                                    .read_allocation()?,
                                input_hadamard_factors: output_embedding_tree
                                    .leaf("input_hadamard_factors")?
                                    .validate(&[model_dim], DataType::I32)?
                                    .read_allocation()?,
                            }) },
                        }
                    },
                    AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                        quantization_spec,
                        adapter_spec: None,
                        incoherence_block_size: Some(block_size),
                        incoherence_processing_mode: IncoherenceProcessingMode::Input,
                        ..
                    }) if block_size == HADAMARD_TRANSFORM_BLOCK_SIZE => {
                        let matrix = WeightMatrix::load(
                            &output_embedding_tree.subtree("quantized"),
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
                            .subtree("incoherence_signs")
                            .leaf("input_signs")?
                            .validate(&[model_dim], DataType::I32)?
                            .read_allocation()?;
                        let kernel = ActivationTransform::input_rht(context, data_type, false)
                            .map_err(EmbeddingError::BackendError)?;

                        UntiedReadout {
                            matrix: Some(matrix),
                            readout: Some(readout_kernel(context, data_type)?),
                            input_hadamard: Some(InputHadamard {
                                factors,
                                kernel,
                            }),
                            i3_s4: None,
                            i3_gemm: None,
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
                            matrix: Some(matrix),
                            readout: Some(readout_kernel(context, data_type)?),
                            input_hadamard: None,
                            i3_s4: None,
                            i3_gemm: None,
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
        let logit_scale = config.logit_scale().unwrap_or(1.0);
        let logit_soft_cap = *config.logit_soft_cap();
        let logit_transform = if logit_scale != 1.0 || logit_soft_cap.is_some() {
            let kernel =
                <B::Kernels as Kernels>::LogitTransformKernel::new(context, data_type, logit_soft_cap.is_some())
                    .map_err(EmbeddingError::BackendError)?;
            let widened_kernel = if data_type != DataType::F32 {
                Some(
                    <B::Kernels as Kernels>::LogitTransformKernel::new(
                        context,
                        DataType::F32,
                        logit_soft_cap.is_some(),
                    )
                    .map_err(EmbeddingError::BackendError)?,
                )
            } else {
                None
            };
            Some(LogitTransform {
                scale: logit_scale,
                soft_cap: logit_soft_cap,
                kernel,
                widened_kernel,
            })
        } else {
            None
        };

        Ok((
            Self {
                tying,
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
        if let EmbeddingTying::Untied {
            output: UntiedReadout {
                i3_gemm: Some(gemm),
                ..
            },
            ..
        } = &self.tying
        {
            let mut transformed =
                encoder.allocate_scratch(input_allocation.size()).map_err(EmbeddingError::BackendError)?;
            gemm.rht.encode_fp(input_allocation, &mut transformed, &gemm.factors, batch_dim, self.model_dim, encoder);
            let mut output = encoder
                .allocate_scratch_for_shape(&[batch_dim, self.vocab_size], output_data_type)
                .map_err(EmbeddingError::BackendError)?;
            if std::env::var("QTIP_RACE_NULL_HEAD").map_or(false, |value| value != "0") {
                encoder.pop_debug_group();
                return Ok(output);
            }
            let arguments: MatmulArguments<'_, '_, '_, B> = MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &transformed,
                    offset: 0,
                },
                b: MatmulB::ScaleSymmetricDequant {
                    b: &gemm.codes,
                    scales: &gemm.scales,
                    mode: gemm.mode,
                    group_size: 64,
                    signed_codes: true,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: MatmulDOps::none(),
                gather_indices: None,
                m: batch_dim,
                n: self.vocab_size,
                k: self.model_dim,
            };
            if native_output {
                gemm.readout.lock().encode(arguments, encoder).map_err(EmbeddingError::BackendError)?;
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
            if let Some(logit_transform) = &self.logit_transform {
                let kernel = if native_output {
                    &logit_transform.kernel
                } else {
                    assert_eq!(output_data_type, DataType::F32);
                    logit_transform.widened_kernel.as_ref().unwrap()
                };
                kernel.encode(
                    &mut output,
                    batch_dim * self.vocab_size,
                    logit_transform.scale,
                    logit_transform.soft_cap.unwrap_or(0.0),
                    encoder,
                );
            }
            encoder.pop_debug_group();
            return Ok(output);
        }
        if let EmbeddingTying::Untied {
            output: UntiedReadout {
                i3_s4: Some(i3_s4),
                ..
            },
            ..
        } = &self.tying
        {
            let mut output = i3_s4
                .kernel
                .encode_i3_s4_readout(
                    I3S4ReadoutArguments {
                        input: input_allocation,
                        codes: &i3_s4.codes,
                        row_scales: &i3_s4.row_scales,
                        ladder_indices: &i3_s4.ladder_indices,
                        ladder: &i3_s4.ladder,
                        input_hadamard_factors: &i3_s4.input_hadamard_factors,
                        batch: batch_dim,
                        vocab_size: self.vocab_size,
                        model_dim: self.model_dim,
                        output_data_type,
                    },
                    encoder,
                )
                .map_err(EmbeddingError::BackendError)?;
            if let Some(logit_transform) = &self.logit_transform {
                let kernel = if native_output {
                    &logit_transform.kernel
                } else {
                    assert_eq!(output_data_type, DataType::F32);
                    logit_transform.widened_kernel.as_ref().unwrap()
                };
                kernel.encode(
                    &mut output,
                    batch_dim * self.vocab_size,
                    logit_transform.scale,
                    logit_transform.soft_cap.unwrap_or(0.0),
                    encoder,
                );
            }
            encoder.pop_debug_group();
            return Ok(output);
        }

        let input_hadamard = self.readout_input_hadamard();
        let mut output_allocation = encoder
            .allocate_scratch_for_shape(&[batch_dim, self.vocab_size], output_data_type)
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

        if let Some(logit_transform) = &self.logit_transform {
            let length = batch_dim * self.vocab_size;
            let kernel = if native_output {
                &logit_transform.kernel
            } else {
                assert_eq!(output_data_type, DataType::F32, "unsupported readout output data type");
                logit_transform.widened_kernel.as_ref().expect("widened logit transform kernel is missing")
            };
            kernel.encode(
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

    /// Hot-band readout: candidates with `token id < hot_rows` are scored with `hot_head`, the rest with `self`
    /// (which must carry an i3/S4 head). Used to emulate a frequency-tiered head for the weaver.
    pub(crate) fn encode_readout_sparse_hybrid(
        &self,
        hot_head: &Embedding<B>,
        hot_rows: u32,
        input: &Allocation<B>,
        token_ids: &Allocation<B>,
        rows: u32,
        ids_per_row: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, EmbeddingError<B>> {
        let cold = self.encode_readout_sparse(input, token_ids, rows, ids_per_row, encoder)?;
        let hot = hot_head.encode_readout_sparse(input, token_ids, rows, ids_per_row, encoder)?;
        let EmbeddingTying::Untied {
            output: UntiedReadout {
                i3_s4: Some(i3_s4),
                ..
            },
            ..
        } = &self.tying
        else {
            return Err(EmbeddingError::UnsupportedConfiguration("hybrid readout needs an i3 cold head".into()));
        };
        assert_eq!(self.data_type, DataType::BF16, "hybrid readout merges bf16 residuals");
        let count = rows * ids_per_row;
        let mut output = encoder
            .allocate_scratch_for_shape(&[rows, ids_per_row], self.data_type)
            .map_err(EmbeddingError::BackendError)?;
        i3_s4.kernel.encode_residual_merge_hot(&hot, &cold, token_ids, &mut output, hot_rows, count, encoder);
        Ok(output)
    }

    /// Per-row candidate readout via the GEMV B-row gather: `out[r][j] == dense[r][token_ids[r][j]]`,
    /// soft-capped when configured, one dispatch. Caller guarantees `token_ids < vocab_size`.
    pub(crate) fn encode_readout_sparse(
        &self,
        input: &Allocation<B>,
        token_ids: &Allocation<B>,
        rows: u32,
        ids_per_row: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, EmbeddingError<B>> {
        encoder.push_debug_group("embedding readout (sparse)");

        assert!(rows > 0 && ids_per_row > 0);
        let force_gemm = std::env::var("QTIP_SPARSE_HEAD").map_or(false, |value| value == "gemm");
        if let EmbeddingTying::Untied {
            output: UntiedReadout {
                i3_gemm: Some(gemm),
                i3_s4,
                ..
            },
            ..
        } = &self.tying
            && (i3_s4.is_none() || force_gemm)
        {
            // symmetric U4/U8 copy of the (possibly tiered) head: gathered GEMV readout, as for asym heads
            let mut transformed = encoder.allocate_scratch(input.size()).map_err(EmbeddingError::BackendError)?;
            gemm.rht.encode_fp(input, &mut transformed, &gemm.factors, rows, self.model_dim, encoder);
            let mut output = encoder
                .allocate_scratch_for_shape(&[rows, ids_per_row], self.data_type)
                .map_err(EmbeddingError::BackendError)?;
            let fuse_soft_cap = match &self.logit_transform {
                Some(logit_transform) if logit_transform.scale != 1.0 => None,
                Some(logit_transform) => logit_transform.soft_cap,
                None => None,
            };
            let arguments: MatmulArguments<'_, '_, '_, B> = MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &transformed,
                    offset: 0,
                },
                b: MatmulB::ScaleSymmetricDequant {
                    b: &gemm.codes,
                    scales: &gemm.scales,
                    mode: gemm.mode,
                    group_size: 64,
                    signed_codes: true,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: MatmulDOps {
                    soft_cap: fuse_soft_cap,
                    ..MatmulDOps::none()
                },
                gather_indices: Some(token_ids),
                m: rows,
                n: ids_per_row,
                k: self.model_dim,
            };
            gemm.readout.lock().encode(arguments, encoder).map_err(EmbeddingError::BackendError)?;
            if let Some(logit_transform) = &self.logit_transform
                && logit_transform.scale != 1.0
            {
                let length = rows * ids_per_row;
                logit_transform.kernel.encode(
                    &mut output,
                    length,
                    logit_transform.scale,
                    logit_transform.soft_cap.unwrap_or(0.0),
                    encoder,
                );
            }
            encoder.pop_debug_group();
            return Ok(output);
        }
        if let EmbeddingTying::Untied {
            output: UntiedReadout {
                i3_s4: Some(i3_s4),
                ..
            },
            ..
        } = &self.tying
        {
            // i3/S4 head: dedicated row-gather kernel (the Hadamard input transform is applied inside)
            let fused_soft_cap = match &self.logit_transform {
                Some(logit_transform) if logit_transform.scale != 1.0 => None,
                Some(logit_transform) => logit_transform.soft_cap,
                None => None,
            };
            let mut output = i3_s4
                .kernel
                .encode_i3_s4_readout_sparse(
                    I3S4SparseReadoutArguments {
                        input,
                        token_ids,
                        codes: &i3_s4.codes,
                        row_scales: &i3_s4.row_scales,
                        ladder_indices: &i3_s4.ladder_indices,
                        ladder: &i3_s4.ladder,
                        input_hadamard_factors: &i3_s4.input_hadamard_factors,
                        rows,
                        ids_per_row,
                        vocab_size: self.vocab_size,
                        model_dim: self.model_dim,
                        output_data_type: self.data_type,
                        soft_cap: fused_soft_cap.unwrap_or(0.0),
                    },
                    encoder,
                )
                .map_err(EmbeddingError::BackendError)?;
            if let Some(logit_transform) = &self.logit_transform
                && logit_transform.scale != 1.0
            {
                let length = rows * ids_per_row;
                logit_transform.kernel.encode(
                    &mut output,
                    length,
                    logit_transform.scale,
                    logit_transform.soft_cap.unwrap_or(0.0),
                    encoder,
                );
            }
            encoder.pop_debug_group();
            return Ok(output);
        }
        let input_hadamard = self.readout_input_hadamard();
        let (matrix, readout) = self.readout_operands();
        let b = matrix.matmul_b();

        let mut output = encoder
            .allocate_scratch_for_shape(&[rows, ids_per_row], self.data_type)
            .map_err(EmbeddingError::BackendError)?;

        let mut rht_input: Option<Allocation<B>> = None;
        let a = match input_hadamard {
            Some(input_hadamard) => {
                let mut transformed = encoder.allocate_scratch(input.size()).map_err(EmbeddingError::BackendError)?;
                input_hadamard.kernel.encode_fp(
                    input,
                    &mut transformed,
                    &input_hadamard.factors,
                    rows,
                    self.model_dim,
                    encoder,
                );
                rht_input.insert(transformed)
            },
            None => input,
        };

        let fuse_soft_cap = match &self.logit_transform {
            Some(logit_transform) if logit_transform.scale != 1.0 => None,
            Some(logit_transform) => logit_transform.soft_cap,
            None => None,
        };
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
                        soft_cap: fuse_soft_cap,
                        ..MatmulDOps::none()
                    },
                    gather_indices: Some(token_ids),
                    m: rows,
                    n: ids_per_row,
                    k: self.model_dim,
                },
                encoder,
            )
            .map_err(EmbeddingError::BackendError)?;

        if let Some(logit_transform) = &self.logit_transform
            && logit_transform.scale != 1.0
        {
            let length = rows * ids_per_row;
            logit_transform.kernel.encode(
                &mut output,
                length,
                logit_transform.scale,
                logit_transform.soft_cap.unwrap_or(0.0),
                encoder,
            );
        }

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

/// Repacks the i3_s4 readout (3-bit levels {-7, -5, ..., 7}, bf16 row scale,
/// 4-bit ladder index per 64 columns into a 16-entry f16 ladder) into the
/// standard symmetric group-64 integer GEMM layout: signed codes (two's
/// complement nibbles, low nibble first, or int8) and bf16 group scales
/// `bf16(row_scale * ladder[index])`.
pub(crate) fn repack_i3_s4_to_symmetric_gemm(
    codes3: &[u8],
    row_scales: &[bf16],
    ladder_indices: &[u8],
    ladder: &[f16],
    vocab: usize,
    dim: usize,
    eight_bit: bool,
) -> (Vec<u8>, Vec<bf16>) {
    let code_stride = dim * 3 / 8;
    repack_levels_to_symmetric_gemm(
        |row, column| i3_level(&codes3[row * code_stride..(row + 1) * code_stride], column),
        row_scales,
        ladder_indices,
        ladder,
        vocab,
        dim,
        eight_bit,
    )
}

/// 3-bit ladder code at `column` of one packed row -> odd level in -7..=7
#[inline]
fn i3_level(row_codes: &[u8], column: usize) -> i32 {
    let bit = column * 3;
    let byte = bit >> 3;
    let shift = bit & 7;
    let mut packed = row_codes[byte] as u32;
    if shift > 5 {
        packed |= (row_codes[byte + 1] as u32) << 8;
    }
    (((packed >> shift) & 7) * 2) as i32 - 7
}

/// Tiered head: rows < `hot_rows` carry signed-nibble INT4 codes (level = nibble - 8), rows in
/// `hot_rows..cold_start` the shipped 3-bit ladder codes, rows >= `cold_start` 2-bit codes (level = 2c - 3).
/// Every band shares the row scale x ladder group multiplier scheme.
pub(crate) fn repack_tiered_to_symmetric_gemm(
    hot_rows: usize,
    cold_start: usize,
    codes4: &[u8],
    codes3: &[u8],
    codes2: &[u8],
    row_scales: &[bf16],
    ladder_indices: &[u8],
    ladder: &[f16],
    vocab: usize,
    dim: usize,
    eight_bit: bool,
) -> (Vec<u8>, Vec<bf16>) {
    assert!(hot_rows <= cold_start && cold_start <= vocab);
    assert_eq!(codes4.len(), hot_rows * dim / 2);
    assert_eq!(codes3.len(), (cold_start - hot_rows) * dim * 3 / 8);
    assert_eq!(codes2.len(), (vocab - cold_start) * dim / 4);
    let stride3 = dim * 3 / 8;
    repack_levels_to_symmetric_gemm(
        |row, column| {
            if row < hot_rows {
                let byte = codes4[row * dim / 2 + column / 2];
                let nibble = if column % 2 == 0 { byte & 15 } else { byte >> 4 };
                nibble as i32 - 8
            } else if row < cold_start {
                let local = row - hot_rows;
                i3_level(&codes3[local * stride3..(local + 1) * stride3], column)
            } else {
                let local = row - cold_start;
                let byte = codes2[local * dim / 4 + column / 4];
                (((byte >> (2 * (column % 4))) & 3) as i32) * 2 - 3
            }
        },
        row_scales,
        ladder_indices,
        ladder,
        vocab,
        dim,
        eight_bit,
    )
}

fn repack_levels_to_symmetric_gemm(
    level_of: impl Fn(usize, usize) -> i32 + Sync,
    row_scales: &[bf16],
    ladder_indices: &[u8],
    ladder: &[f16],
    vocab: usize,
    dim: usize,
    eight_bit: bool,
) -> (Vec<u8>, Vec<bf16>) {
    let level_of = &level_of;
    let ladder_stride = dim / 128;
    let groups = dim / 64;
    let out_stride = if eight_bit {
        dim
    } else {
        dim / 2
    };
    let mut codes = vec![0u8; vocab * out_stride];
    let mut scales = vec![bf16::ZERO; vocab * groups];
    let threads = std::thread::available_parallelism().map_or(8, |count| count.get()).clamp(1, 16);
    let rows_per_chunk = vocab.div_ceil(threads);
    std::thread::scope(|scope| {
        for (chunk_index, (codes_chunk, scales_chunk)) in
            codes.chunks_mut(rows_per_chunk * out_stride).zip(scales.chunks_mut(rows_per_chunk * groups)).enumerate()
        {
            let row_start = chunk_index * rows_per_chunk;
            scope.spawn(move || {
                for (local_row, (row_codes, row_scales_out)) in
                    codes_chunk.chunks_mut(out_stride).zip(scales_chunk.chunks_mut(groups)).enumerate()
                {
                    let row = row_start + local_row;
                    let ladders = &ladder_indices[row * ladder_stride..(row + 1) * ladder_stride];
                    let row_scale = row_scales[row].to_f32();
                    for (group, scale) in row_scales_out.iter_mut().enumerate() {
                        let packed = ladders[group / 2];
                        let index = if group % 2 == 0 {
                            packed & 15
                        } else {
                            packed >> 4
                        };
                        *scale = bf16::from_f32(row_scale * ladder[index as usize].to_f32());
                    }
                    for column in 0..dim {
                        let level = level_of(row, column);
                        debug_assert!((-8..=7).contains(&level), "level {level} does not fit a signed nibble");
                        if eight_bit {
                            row_codes[column] = level as i8 as u8;
                        } else {
                            let nibble = (level as u8) & 0xF;
                            if column % 2 == 0 {
                                row_codes[column / 2] = nibble;
                            } else {
                                row_codes[column / 2] |= nibble << 4;
                            }
                        }
                    }
                }
            });
        }
    });
    (codes, scales)
}
