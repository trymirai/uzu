use parking_lot::Mutex;
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
        readout: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
    },
    Quantized {
        weights: Allocation<B>,
        scales: Allocation<B>,
        zero_points_or_biases: Option<Allocation<B>>,
        quantization_method: QuantizationMethod,
        output_hadamard_factors: Option<Allocation<B>>,
        lookup: <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel,
        readout: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
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
        readout: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
    },
    Quantized {
        weights: Allocation<B>,
        scales: Allocation<B>,
        zero_points_or_biases: Option<Allocation<B>>,
        readout: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
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
    vocab_size: u32,
    model_dim: u32,
}

struct LogitSoftCap<B: Backend> {
    value: f32,
    kernel: <B::Kernels as Kernels>::LogitSoftCapKernel,
}

impl<B: Backend> Embedding<B> {
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
                        let readout_kernel =
                            <B::Kernels as Kernels>::MatmulKernel::new(context, data_type, data_type, data_type)
                                .map_err(EmbeddingError::BackendError)?;
                        let readout = Mutex::new(readout_kernel);

                        (
                            TiedEmbeddingType::FullPrecision {
                                weights,
                                lookup,
                                readout,
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
                        let (readout, readout_config) = quantized_readout(
                            context,
                            data_type,
                            embedding_quantization_mode,
                            quantization_method,
                            group_size,
                        )?;

                        (
                            TiedEmbeddingType::Quantized {
                                weights,
                                scales,
                                zero_points_or_biases,
                                quantization_method,
                                output_hadamard_factors: None,
                                lookup,
                                readout,
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
                        let (readout, readout_config) = quantized_readout(
                            context,
                            data_type,
                            embedding_quantization_mode,
                            quantization_method,
                            group_size,
                        )?;

                        (
                            TiedEmbeddingType::Quantized {
                                weights,
                                scales,
                                zero_points_or_biases,
                                quantization_method,
                                output_hadamard_factors,
                                lookup,
                                readout,
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
                        let readout_kernel =
                            <B::Kernels as Kernels>::MatmulKernel::new(context, data_type, data_type, data_type)
                                .map_err(EmbeddingError::BackendError)?;
                        let readout = Mutex::new(readout_kernel);

                        UntiedEmbeddingReadoutType::FullPrecision {
                            weights,
                            readout,
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
                        let (readout, readout_config) = quantized_readout(
                            context,
                            data_type,
                            embedding_quantization_mode,
                            quantization_method,
                            group_size,
                        )?;

                        UntiedEmbeddingReadoutType::Quantized {
                            weights,
                            scales,
                            zero_points_or_biases,
                            readout,
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
                        readout: _,
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
                        readout: _,
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
        batch_dim: usize,
        input_allocation: &Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, EmbeddingError<B>> {
        assert!(batch_dim > 0, "Embedding readout requires at least one row");
        let mut output_allocation = encoder
            .allocate_scratch(size_for_shape(&[batch_dim, self.vocab_size as usize], self.data_type))
            .map_err(EmbeddingError::BackendError)?;

        match &self.tying {
            EmbeddingTying::Tied {
                ty:
                    TiedEmbeddingType::FullPrecision {
                        weights,
                        lookup: _,
                        readout,
                    },
            }
            | EmbeddingTying::Untied {
                input_ty: _,
                output_ty:
                    UntiedEmbeddingReadoutType::FullPrecision {
                        weights,
                        readout,
                    },
            } => {
                readout
                    .lock()
                    .encode(
                        MatmulArguments {
                            a: input_allocation,
                            a_offset: 0,
                            b: MatmulB::FullPrecision {
                                b: weights,
                            },
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
            },
            EmbeddingTying::Tied {
                ty:
                    TiedEmbeddingType::Quantized {
                        weights,
                        scales,
                        zero_points_or_biases,
                        quantization_method: _,
                        output_hadamard_factors: _,
                        lookup: _,
                        readout,
                        readout_config,
                    },
            }
            | EmbeddingTying::Untied {
                input_ty: _,
                output_ty:
                    UntiedEmbeddingReadoutType::Quantized {
                        weights,
                        scales,
                        zero_points_or_biases,
                        readout,
                        readout_config,
                    },
            } => {
                let b: MatmulB<'_, B> = match readout_config.method {
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
                readout
                    .lock()
                    .encode(
                        MatmulArguments {
                            a: input_allocation,
                            a_offset: 0,
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
            },
        };

        if let Some(logit_soft_cap) = &self.logit_soft_cap {
            logit_soft_cap.kernel.encode(
                &mut output_allocation,
                (batch_dim * self.vocab_size as usize) as u32,
                logit_soft_cap.value,
                encoder,
            );
        }

        Ok(output_allocation)
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

fn quantized_readout<B: Backend>(
    context: &B::Context,
    data_type: DataType,
    mode: QuantizationMode,
    method: QuantizationMethod,
    group_size: usize,
) -> Result<(Mutex<<B::Kernels as Kernels>::MatmulKernel>, ReadoutQuantConfig), EmbeddingError<B>> {
    let readout = <B::Kernels as Kernels>::MatmulKernel::new(context, data_type, data_type, data_type)
        .map_err(EmbeddingError::BackendError)?;
    Ok((
        Mutex::new(readout),
        ReadoutQuantConfig {
            method,
            mode,
            group_size: group_size as u32,
        },
    ))
}
