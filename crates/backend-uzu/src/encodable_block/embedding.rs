use std::cell::RefCell;

use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::{QuantizationMethod, QuantizationMode},
        kernel::{
            FullPrecisionEmbeddingLookupKernel, LogitSoftCapKernel, QuantizedEmbeddingLookupKernel,
            matmul::{MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
        },
    },
    config::{
        embedding::AnyEmbeddingConfig,
        weight_matrix::{
            AnyWeightMatrixSpec, Layout,
            full_precision_spec::FullPrecisionSpec,
            hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
            int_spec::IntSpec,
            mlx_spec::MLXSpec,
        },
    },
    data_type::DataType,
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
}

struct ReadoutQuantConfig {
    method: QuantizationMethod,
    mode: QuantizationMode,
    group_size: u32,
}

enum TiedEmbeddingType<B: Backend> {
    FullPrecision {
        weights: Allocation<B>,
        lookup: <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel,
    },
    Quantized {
        weights: Allocation<B>,
        scales: Allocation<B>,
        zero_points_or_biases: Option<Allocation<B>>,
        quantization_method: QuantizationMethod,
        output_hadamard_factors: Option<Allocation<B>>,
        lookup: <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel,
        readout_config: ReadoutQuantConfig,
    },
}

enum UntiedEmbeddingLookupType<B: Backend> {
    FullPrecision {
        weights: Allocation<B>,
        lookup: <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel,
    },
    Quantized {
        weights: Allocation<B>,
        scales: Allocation<B>,
        zero_points_or_biases: Option<Allocation<B>>,
        quantization_method: QuantizationMethod,
        output_hadamard_factors: Option<Allocation<B>>,
        lookup: <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel,
    },
}

enum UntiedEmbeddingReadoutType<B: Backend> {
    FullPrecision {
        weights: Allocation<B>,
    },
    Quantized {
        weights: Allocation<B>,
        scales: Allocation<B>,
        zero_points_or_biases: Option<Allocation<B>>,
        readout_config: ReadoutQuantConfig,
    },
}

enum EmbeddingTying<B: Backend> {
    Tied {
        ty: TiedEmbeddingType<B>,
    },
    Untied {
        input_ty: UntiedEmbeddingLookupType<B>,
        output_ty: UntiedEmbeddingReadoutType<B>,
    },
}

pub struct Embedding<B: Backend> {
    tying: EmbeddingTying<B>,
    input_scale: f32,
    data_type: DataType,
    logit_soft_cap: Option<LogitSoftCap<B>>,
    readouts: Box<[(DataType, RefCell<<B::Kernels as Kernels>::MatmulKernel>)]>,
    vocab_size: u32,
    model_dim: u32,
}

struct LogitSoftCap<B: Backend> {
    value: f32,
    kernels: Box<[(DataType, <B::Kernels as Kernels>::LogitSoftCapKernel)]>,
}

impl<B: Backend> LogitSoftCap<B> {
    fn kernel(
        &self,
        output_data_type: DataType,
    ) -> &<B::Kernels as Kernels>::LogitSoftCapKernel {
        self.kernels
            .iter()
            .find(|(data_type, _)| *data_type == output_data_type)
            .map(|(_, kernel)| kernel)
            .expect("readout output dtype validated at entry")
    }
}

// TODO: embedding lookup dtype (u64) should match sampling (u32)

impl<B: Backend> Embedding<B> {
    pub(crate) fn vocab_size(&self) -> usize {
        self.vocab_size as usize
    }

    pub(crate) fn model_dim(&self) -> usize {
        self.model_dim as usize
    }

    pub(crate) fn data_type(&self) -> DataType {
        self.data_type
    }

    fn readout_kernel(
        &self,
        output_data_type: DataType,
    ) -> &RefCell<<B::Kernels as Kernels>::MatmulKernel> {
        self.readouts
            .iter()
            .find(|(data_type, _)| *data_type == output_data_type)
            .map(|(_, kernel)| kernel)
            .expect("readout output dtype validated at entry")
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

                let (ty, readout_input_hadamard_factors) = match embedding_spec {
                    AnyWeightMatrixSpec::FullPrecisionSpec(FullPrecisionSpec {
                        layout: Layout::InputOutput,
                        ..
                    }) => {
                        let weights = embedding_tree
                            .leaf("weights")?
                            .validate(&[vocab_size as usize, model_dim as usize], data_type)?
                            .read_allocation()?;

                        let lookup =
                            <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel::new(context, data_type)
                                .map_err(EmbeddingError::BackendError)?;

                        (
                            TiedEmbeddingType::FullPrecision {
                                weights,
                                lookup,
                            },
                            None,
                        )
                    },
                    spec @ (AnyWeightMatrixSpec::MLXSpec(_) | AnyWeightMatrixSpec::IntSpec(_)) => {
                        let (embedding_quantization_mode, group_size, quantization_method) =
                            input_quantization_from_spec(spec)?;
                        let (weights, scales, zero_points_or_biases) = load_quantized_embedding_parts(
                            &embedding_tree,
                            vocab_size as usize,
                            model_dim as usize,
                            data_type,
                            embedding_quantization_mode,
                            quantization_method,
                            group_size,
                        )?;

                        let lookup = <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel::new(
                            context,
                            data_type,
                            group_size as u32,
                            embedding_quantization_mode,
                            quantization_method,
                            false,
                        )
                        .map_err(EmbeddingError::BackendError)?;
                        let readout_config = ReadoutQuantConfig {
                            method: quantization_method,
                            mode: embedding_quantization_mode,
                            group_size: group_size as u32,
                        };

                        (
                            TiedEmbeddingType::Quantized {
                                weights,
                                scales,
                                zero_points_or_biases,
                                quantization_method,
                                output_hadamard_factors: None,
                                lookup,
                                readout_config,
                            },
                            None,
                        )
                    },
                    AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                        quantization_spec,
                        adapter_spec: None,
                        incoherence_block_size: Some(32),
                        incoherence_processing_mode: IncoherenceProcessingMode::Output,
                        ..
                    }) => {
                        let (embedding_quantization_mode, group_size, quantization_method) =
                            input_quantization_from_spec(*quantization_spec)?;
                        let quantized_tree = embedding_tree.subtree("quantized")?;
                        let (weights, scales, zero_points_or_biases) = load_quantized_embedding_parts(
                            &quantized_tree,
                            vocab_size as usize,
                            model_dim as usize,
                            data_type,
                            embedding_quantization_mode,
                            quantization_method,
                            group_size,
                        )?;

                        let incoherence_signs_tree = embedding_tree.subtree("incoherence_signs")?;
                        let output_hadamard_factors = Some(
                            incoherence_signs_tree
                                .leaf("output_signs")?
                                .validate(&[model_dim as usize], DataType::I32)?
                                .read_allocation()?,
                        );
                        let readout_input_hadamard_factors = Some(
                            incoherence_signs_tree
                                .leaf("output_signs")?
                                .validate(&[model_dim as usize], DataType::I32)?
                                .read_allocation()?,
                        );
                        let lookup = <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel::new(
                            context,
                            data_type,
                            group_size as u32,
                            embedding_quantization_mode,
                            quantization_method,
                            true,
                        )
                        .map_err(EmbeddingError::BackendError)?;
                        let readout_config = ReadoutQuantConfig {
                            method: quantization_method,
                            mode: embedding_quantization_mode,
                            group_size: group_size as u32,
                        };

                        (
                            TiedEmbeddingType::Quantized {
                                weights,
                                scales,
                                zero_points_or_biases,
                                quantization_method,
                                output_hadamard_factors,
                                lookup,
                                readout_config,
                            },
                            readout_input_hadamard_factors,
                        )
                    },
                    spec => return Err(EmbeddingError::UnsupportedConfiguration(format!("{spec:?}"))),
                };

                (
                    EmbeddingTying::Tied {
                        ty,
                    },
                    readout_input_hadamard_factors,
                )
            },
            AnyEmbeddingConfig::UntiedEmbeddingConfig(_) => {
                let input_embedding_tree = parameter_tree.subtree("input_embedding")?;
                let input_embedding_spec = input_embedding_tree.metadata::<AnyWeightMatrixSpec>("spec")?;

                let input_ty = match input_embedding_spec {
                    AnyWeightMatrixSpec::FullPrecisionSpec(FullPrecisionSpec {
                        layout: Layout::InputOutput,
                        ..
                    }) => {
                        let weights = input_embedding_tree
                            .leaf("weights")?
                            .validate(&[vocab_size as usize, model_dim as usize], data_type)?
                            .read_allocation()?;

                        let lookup =
                            <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel::new(context, data_type)
                                .map_err(EmbeddingError::BackendError)?;

                        UntiedEmbeddingLookupType::FullPrecision {
                            weights,
                            lookup,
                        }
                    },
                    spec @ (AnyWeightMatrixSpec::MLXSpec(_) | AnyWeightMatrixSpec::IntSpec(_)) => {
                        let (embedding_quantization_mode, group_size, quantization_method) =
                            input_quantization_from_spec(spec)?;
                        let (weights, scales, zero_points_or_biases) = load_quantized_embedding_parts(
                            &input_embedding_tree,
                            vocab_size as usize,
                            model_dim as usize,
                            data_type,
                            embedding_quantization_mode,
                            quantization_method,
                            group_size,
                        )?;

                        let lookup = <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel::new(
                            context,
                            data_type,
                            group_size as u32,
                            embedding_quantization_mode,
                            quantization_method,
                            false,
                        )
                        .map_err(EmbeddingError::BackendError)?;

                        UntiedEmbeddingLookupType::Quantized {
                            weights,
                            scales,
                            zero_points_or_biases,
                            quantization_method,
                            output_hadamard_factors: None,
                            lookup,
                        }
                    },
                    spec => return Err(EmbeddingError::UnsupportedConfiguration(format!("{spec:?}"))),
                };

                let output_embedding_tree = parameter_tree.subtree("output_embedding")?;
                let output_embedding_spec = output_embedding_tree.metadata::<AnyWeightMatrixSpec>("spec")?;

                let output_ty = match output_embedding_spec {
                    AnyWeightMatrixSpec::FullPrecisionSpec(FullPrecisionSpec {
                        layout: Layout::OutputInput,
                        ..
                    }) => {
                        let weights = output_embedding_tree
                            .leaf("weights")?
                            .validate(&[vocab_size as usize, model_dim as usize], data_type)?
                            .read_allocation()?;
                        UntiedEmbeddingReadoutType::FullPrecision {
                            weights,
                        }
                    },
                    spec @ (AnyWeightMatrixSpec::MLXSpec(_) | AnyWeightMatrixSpec::IntSpec(_)) => {
                        let (embedding_quantization_mode, group_size, quantization_method) =
                            output_quantization_from_spec(spec)?;
                        let (weights, scales, zero_points_or_biases) = load_quantized_embedding_parts(
                            &output_embedding_tree,
                            vocab_size as usize,
                            model_dim as usize,
                            data_type,
                            embedding_quantization_mode,
                            quantization_method,
                            group_size,
                        )?;
                        let readout_config = ReadoutQuantConfig {
                            method: quantization_method,
                            mode: embedding_quantization_mode,
                            group_size: group_size as u32,
                        };

                        UntiedEmbeddingReadoutType::Quantized {
                            weights,
                            scales,
                            zero_points_or_biases,
                            readout_config,
                        }
                    },
                    spec => return Err(EmbeddingError::UnsupportedConfiguration(format!("{spec:?}"))),
                };

                (
                    EmbeddingTying::Untied {
                        input_ty,
                        output_ty,
                    },
                    None,
                )
            },
        };

        let output_data_types: &[DataType] = if data_type == DataType::F32 {
            &[DataType::F32]
        } else {
            &[data_type, DataType::F32]
        };
        let input_scale = config.input_scale().unwrap_or(1.0);
        let logit_soft_cap = if let Some(value) = *config.logit_soft_cap() {
            let kernels = output_data_types
                .iter()
                .map(|&output_data_type| {
                    <B::Kernels as Kernels>::LogitSoftCapKernel::new(context, output_data_type)
                        .map(|kernel| (output_data_type, kernel))
                        .map_err(EmbeddingError::BackendError)
                })
                .collect::<Result<Box<[_]>, _>>()?;
            Some(LogitSoftCap {
                value,
                kernels,
            })
        } else {
            None
        };
        let readouts = output_data_types
            .iter()
            .map(|&output_data_type| {
                <B::Kernels as Kernels>::MatmulKernel::new(context, data_type, data_type, output_data_type)
                    .map(|kernel| (output_data_type, RefCell::new(kernel)))
                    .map_err(EmbeddingError::BackendError)
            })
            .collect::<Result<Box<[_]>, _>>()?;

        Ok((
            Self {
                tying,
                input_scale,
                data_type,
                logit_soft_cap,
                readouts,
                vocab_size,
                model_dim,
            },
            readout_input_hadamard_factors,
        ))
    }

    pub fn encode_lookup(
        &self,
        token_ids: &Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, EmbeddingError<B>> {
        let mut output = encoder
            .allocate_scratch(size_for_shape(&[batch_dim, self.model_dim as usize], self.data_type))
            .map_err(EmbeddingError::BackendError)?;
        let batch_dim = batch_dim as u32;

        match &self.tying {
            EmbeddingTying::Tied {
                ty:
                    TiedEmbeddingType::FullPrecision {
                        weights,
                        lookup,
                    },
            }
            | EmbeddingTying::Untied {
                input_ty:
                    UntiedEmbeddingLookupType::FullPrecision {
                        weights,
                        lookup,
                    },
                output_ty: _,
            } => lookup.encode(
                token_ids,
                weights,
                &mut output,
                batch_dim,
                self.vocab_size,
                self.model_dim,
                self.input_scale,
                encoder,
            ),
            EmbeddingTying::Tied {
                ty:
                    TiedEmbeddingType::Quantized {
                        weights,
                        scales,
                        zero_points_or_biases,
                        quantization_method,
                        output_hadamard_factors,
                        lookup,
                        readout_config: _,
                    },
            }
            | EmbeddingTying::Untied {
                input_ty:
                    UntiedEmbeddingLookupType::Quantized {
                        weights,
                        scales,
                        zero_points_or_biases,
                        quantization_method,
                        output_hadamard_factors,
                        lookup,
                    },
                output_ty: _,
            } => {
                let (zero_points, biases) = match quantization_method {
                    QuantizationMethod::ScaleBias => (None, zero_points_or_biases.as_ref()),
                    QuantizationMethod::ScaleZeroPoint => (zero_points_or_biases.as_ref(), None),
                    QuantizationMethod::ScaleSymmetric => (None, None),
                };
                lookup.encode(
                    token_ids,
                    weights,
                    scales,
                    zero_points,
                    biases,
                    &mut output,
                    output_hadamard_factors.as_ref(),
                    batch_dim,
                    self.vocab_size,
                    self.model_dim,
                    self.input_scale,
                    encoder,
                );
            },
        };

        Ok(output)
    }

    pub fn encode_readout(
        &self,
        row_offset: usize,
        batch_dim: usize,
        input_allocation: &Allocation<B>,
        output_data_type: DataType,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, EmbeddingError<B>> {
        assert!(batch_dim > 0, "Embedding readout requires at least one row");
        assert!(
            output_data_type == self.data_type || output_data_type == DataType::F32,
            "readout output dtype must be the model dtype ({:?}) or F32, got {:?}",
            self.data_type,
            output_data_type,
        );
        let native_output = output_data_type == self.data_type;
        let a_offset = row_offset * self.model_dim as usize * self.data_type.size_in_bytes();
        let mut output_allocation = encoder
            .allocate_scratch(size_for_shape(&[batch_dim, self.vocab_size as usize], output_data_type))
            .map_err(EmbeddingError::BackendError)?;

        let b: MatmulB<'_, B> = match &self.tying {
            EmbeddingTying::Tied {
                ty:
                    TiedEmbeddingType::FullPrecision {
                        weights,
                        ..
                    },
            }
            | EmbeddingTying::Untied {
                output_ty: UntiedEmbeddingReadoutType::FullPrecision {
                    weights,
                },
                ..
            } => MatmulB::FullPrecision {
                b: weights,
            },
            EmbeddingTying::Tied {
                ty:
                    TiedEmbeddingType::Quantized {
                        weights,
                        scales,
                        zero_points_or_biases,
                        readout_config,
                        ..
                    },
            }
            | EmbeddingTying::Untied {
                output_ty:
                    UntiedEmbeddingReadoutType::Quantized {
                        weights,
                        scales,
                        zero_points_or_biases,
                        readout_config,
                    },
                ..
            } => {
                if !native_output {
                    return Err(EmbeddingError::UnsupportedConfiguration(
                        "readout requires full-precision weights".to_string(),
                    ));
                }
                let b = match readout_config.method {
                    QuantizationMethod::ScaleBias => MatmulB::ScaleBiasDequant {
                        b: weights,
                        scales,
                        biases: zero_points_or_biases.as_ref().expect("ScaleBias quantization requires biases"),
                        mode: readout_config.mode,
                        group_size: readout_config.group_size,
                    },
                    QuantizationMethod::ScaleZeroPoint => MatmulB::ScaleZeroPointDequant {
                        b: weights,
                        scales,
                        zero_points: zero_points_or_biases
                            .as_ref()
                            .expect("ScaleZeroPoint quantization requires zero_points"),
                        mode: readout_config.mode,
                        group_size: readout_config.group_size,
                    },
                    QuantizationMethod::ScaleSymmetric => MatmulB::ScaleSymmetricDequant {
                        b: weights,
                        scales,
                        mode: readout_config.mode,
                        group_size: readout_config.group_size,
                    },
                };
                b
            },
        };

        let mut kernel = self.readout_kernel(output_data_type).borrow_mut();
        kernel
            .encode(
                MatmulArguments {
                    a: input_allocation,
                    a_offset,
                    b,
                    b_leading_dimension: None,
                    b_transpose: true,
                    d: &mut output_allocation,
                    d_transform: MatmulDOps::none(),
                    m: batch_dim as u32,
                    n: self.vocab_size,
                    k: self.model_dim,
                },
                encoder,
            )
            .map_err(EmbeddingError::BackendError)?;

        if let Some(logit_soft_cap) = &self.logit_soft_cap {
            logit_soft_cap.kernel(output_data_type).encode(
                &mut output_allocation,
                (batch_dim * self.vocab_size as usize) as u32,
                logit_soft_cap.value,
                encoder,
            );
        }

        Ok(output_allocation)
    }

    pub(crate) fn encode_readout_sparse(
        &self,
        input: &Allocation<B>,
        token_ids: &Allocation<B>,
        rows: usize,
        ids_per_row: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, EmbeddingError<B>> {
        assert!(rows > 0 && ids_per_row > 0);
        let (weights, lookup) = match &self.tying {
            EmbeddingTying::Tied {
                ty:
                    TiedEmbeddingType::FullPrecision {
                        weights,
                        lookup,
                        ..
                    },
            } => (weights, lookup),
            EmbeddingTying::Untied {
                input_ty:
                    UntiedEmbeddingLookupType::FullPrecision {
                        lookup,
                        ..
                    },
                output_ty:
                    UntiedEmbeddingReadoutType::FullPrecision {
                        weights,
                        ..
                    },
            } => (weights, lookup),
            _ => {
                return Err(EmbeddingError::UnsupportedConfiguration(
                    "sparse readout currently requires full-precision embedding weights".to_string(),
                ));
            },
        };
        let input_dim = self.model_dim as usize;
        let mut output = encoder
            .allocate_scratch(size_for_shape(&[rows, ids_per_row], DataType::F32))
            .map_err(EmbeddingError::BackendError)?;
        let row_bytes = input_dim * self.data_type.size_in_bytes();
        let ids_row_bytes = ids_per_row * DataType::U64.size_in_bytes();
        let output_row_bytes = ids_per_row * DataType::F32.size_in_bytes();

        // Reused single-row scratch: one rows*n gather buffer falls off the
        // scratch pool's cheap path (~10ms/round); in-order encoding makes reuse safe.
        let mut gathered = encoder
            .allocate_scratch(size_for_shape(&[ids_per_row, input_dim], self.data_type))
            .map_err(EmbeddingError::BackendError)?;
        // MatmulArguments has no output offset, so rows land via copy.
        let mut row_output = encoder
            .allocate_scratch(size_for_shape(&[1, ids_per_row], DataType::F32))
            .map_err(EmbeddingError::BackendError)?;
        let mut readout = self.readout_kernel(DataType::F32).borrow_mut();
        for row in 0..rows {
            lookup.encode(
                (token_ids, row * ids_row_bytes),
                weights,
                &mut gathered,
                ids_per_row as u32,
                self.vocab_size,
                input_dim as u32,
                1.0,
                encoder,
            );
            readout
                .encode(
                    MatmulArguments {
                        a: input,
                        a_offset: row * row_bytes,
                        b: MatmulB::FullPrecision {
                            b: &gathered,
                        },
                        b_leading_dimension: None,
                        b_transpose: true,
                        d: &mut row_output,
                        d_transform: MatmulDOps::none(),
                        m: 1,
                        n: ids_per_row as u32,
                        k: input_dim as u32,
                    },
                    encoder,
                )
                .map_err(EmbeddingError::BackendError)?;
            encoder.encode_copy(&row_output, .., &mut output, row * output_row_bytes..(row + 1) * output_row_bytes);
        }
        Ok(output)
    }
}

fn input_quantization_from_spec<B: Backend>(
    spec: AnyWeightMatrixSpec
) -> Result<(QuantizationMode, usize, QuantizationMethod), EmbeddingError<B>> {
    match spec {
        AnyWeightMatrixSpec::MLXSpec(MLXSpec {
            bits,
            group_size,
            layout: Layout::InputOutput,
            ..
        }) => quantization_mode(bits, group_size, QuantizationMethod::ScaleBias),
        AnyWeightMatrixSpec::IntSpec(IntSpec {
            bits,
            group_size,
            is_symmetric: false,
            layout: Layout::InputOutput,
            ..
        }) => quantization_mode(bits, group_size, QuantizationMethod::ScaleZeroPoint),
        AnyWeightMatrixSpec::IntSpec(IntSpec {
            bits,
            group_size,
            is_symmetric: true,
            layout: Layout::InputOutput,
            ..
        }) => quantization_mode(bits, group_size, QuantizationMethod::ScaleSymmetric),
        spec => Err(EmbeddingError::UnsupportedConfiguration(format!("{spec:?}"))),
    }
}

fn output_quantization_from_spec<B: Backend>(
    spec: AnyWeightMatrixSpec
) -> Result<(QuantizationMode, usize, QuantizationMethod), EmbeddingError<B>> {
    match spec {
        AnyWeightMatrixSpec::MLXSpec(MLXSpec {
            bits,
            group_size,
            layout: Layout::OutputInput,
            ..
        }) => quantization_mode(bits, group_size, QuantizationMethod::ScaleBias),
        AnyWeightMatrixSpec::IntSpec(IntSpec {
            bits,
            group_size,
            is_symmetric: false,
            layout: Layout::OutputInput,
            ..
        }) => quantization_mode(bits, group_size, QuantizationMethod::ScaleZeroPoint),
        AnyWeightMatrixSpec::IntSpec(IntSpec {
            bits,
            group_size,
            is_symmetric: true,
            layout: Layout::OutputInput,
            ..
        }) => quantization_mode(bits, group_size, QuantizationMethod::ScaleSymmetric),
        spec => Err(EmbeddingError::UnsupportedConfiguration(format!("{spec:?}"))),
    }
}

fn quantization_mode<B: Backend>(
    bits: u32,
    group_size: usize,
    method: QuantizationMethod,
) -> Result<(QuantizationMode, usize, QuantizationMethod), EmbeddingError<B>> {
    let mode = match bits {
        4 => QuantizationMode::U4,
        8 => QuantizationMode::U8,
        _ => {
            return Err(EmbeddingError::UnsupportedConfiguration(format!(
                "{method:?} bits={bits}, group_size={group_size}"
            )));
        },
    };
    Ok((mode, group_size, method))
}

fn load_quantized_embedding_parts<B: Backend>(
    tree: &ParameterTree<B>,
    vocab_size: usize,
    model_dim: usize,
    data_type: DataType,
    quantization_mode: QuantizationMode,
    quantization_method: QuantizationMethod,
    group_size: usize,
) -> Result<(Allocation<B>, Allocation<B>, Option<Allocation<B>>), EmbeddingError<B>> {
    let packing_divisor = quantization_mode.packing_divisor();
    let storage_data_type = quantization_mode.storage_type();
    let num_groups = model_dim.div_ceil(group_size);

    let weights = tree
        .leaf("weights")?
        .validate(&[vocab_size, model_dim / packing_divisor], storage_data_type)?
        .read_allocation()?;
    let scales = tree.leaf("scales")?.validate(&[vocab_size, num_groups], data_type)?.read_allocation()?;
    let zero_points_or_biases = match quantization_method {
        QuantizationMethod::ScaleBias => {
            Some(tree.leaf("biases")?.validate(&[vocab_size, num_groups], data_type)?.read_allocation()?)
        },
        QuantizationMethod::ScaleZeroPoint => {
            let expected_zero_points_entries = num_groups.div_ceil(packing_divisor);
            Some(
                tree.leaf("zero_points")?
                    .validate(&[vocab_size, expected_zero_points_entries], storage_data_type)?
                    .read_allocation()?,
            )
        },
        QuantizationMethod::ScaleSymmetric => None,
    };

    Ok((weights, scales, zero_points_or_biases))
}
