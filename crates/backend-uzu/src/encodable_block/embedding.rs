use std::cell::RefCell;

use thiserror::Error;

use crate::{
    DataType,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::QuantizationMethod,
        kernel::{
            FullPrecisionEmbeddingLookupKernel, ManualKernels, QuantizedEmbeddingLookupKernel,
            matmul::{MatmulArgumentC, MatmulArguments, MatmulError, MatmulKernel},
            quant_matmul::{
                QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulError,
            },
        },
    },
    config::EmbeddingConfig,
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum EmbeddingError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Matmul error: {0}")]
    MatmulError(#[from] MatmulError<B>),
    #[error("QuantizedMatmul error: {0}")]
    QuantizedMatmulError(#[from] QuantizedMatmulError<B>),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Unsupported configuration: {0}")]
    UnsupportedConfiguration(String),
}

enum TiedEmbeddingType<B: Backend> {
    FullPrecision {
        weights: Allocation<B>,
        lookup: <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel,
        readout: RefCell<<B::Kernels as ManualKernels>::MatmulKernel>,
    },
    Quantized {
        weights: Allocation<B>,
        scales: Allocation<B>,
        biases: Allocation<B>,
        lookup: <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel,
        readout: RefCell<<B::Kernels as ManualKernels>::MatmulKernel>,
        readout_config: QuantizedMatmulConfiguration,
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
        biases: Allocation<B>,
        lookup: <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel,
    },
}

enum UntiedEmbeddingReadoutType<B: Backend> {
    FullPrecision {
        weights: Allocation<B>,
        readout: RefCell<<B::Kernels as ManualKernels>::MatmulKernel>,
    },
    Quantized {
        weights: Allocation<B>,
        scales: Allocation<B>,
        biases: Allocation<B>,
        readout: RefCell<<B::Kernels as ManualKernels>::MatmulKernel>,
        readout_config: QuantizedMatmulConfiguration,
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
    activation_data_type: DataType,
    vocab_size: u32,
    model_dim: u32,
}

struct UntiedWeightKeys {
    lookup_weights: &'static str,
    lookup_scales: &'static str,
    lookup_biases: &'static str,
    readout_weights: &'static str,
    readout_scales: &'static str,
    readout_biases: &'static str,
}

const SHARED_TREE_UNTIED_KEYS: UntiedWeightKeys = UntiedWeightKeys {
    lookup_weights: "input_weights",
    lookup_scales: "input_scales",
    lookup_biases: "input_biases",
    readout_weights: "output_weights",
    readout_scales: "output_scales",
    readout_biases: "output_biases",
};

#[cfg(metal_backend)]
const SPLIT_TREE_UNTIED_KEYS: UntiedWeightKeys = UntiedWeightKeys {
    lookup_weights: "weights",
    lookup_scales: "scales",
    lookup_biases: "biases",
    readout_weights: "weights",
    readout_scales: "scales",
    readout_biases: "biases",
};

impl<B: Backend> Embedding<B> {
    pub fn new(
        context: &B::Context,
        vocab_size: u32,
        model_dim: u32,
        config: &EmbeddingConfig,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, EmbeddingError<B>> {
        Self::new_with_weight_trees(
            context,
            vocab_size,
            model_dim,
            config,
            parameter_tree,
            parameter_tree,
            &SHARED_TREE_UNTIED_KEYS,
        )
    }

    #[cfg(metal_backend)]
    pub(crate) fn new_with_lookup_and_readout_trees(
        context: &B::Context,
        vocab_size: u32,
        model_dim: u32,
        config: &EmbeddingConfig,
        lookup_tree: &ParameterTree<B::Context>,
        readout_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, EmbeddingError<B>> {
        Self::new_with_weight_trees(
            context,
            vocab_size,
            model_dim,
            config,
            lookup_tree,
            readout_tree,
            &SPLIT_TREE_UNTIED_KEYS,
        )
    }

    fn new_with_weight_trees(
        context: &B::Context,
        vocab_size: u32,
        model_dim: u32,
        config: &EmbeddingConfig,
        lookup_tree: &ParameterTree<B::Context>,
        readout_tree: &ParameterTree<B::Context>,
        untied_keys: &UntiedWeightKeys,
    ) -> Result<Self, EmbeddingError<B>> {
        let common = config.common();

        let tying = match config {
            EmbeddingConfig::Tied {
                common: _,
                precision,
            } => {
                let data_type = (*precision).into();

                let weights_leaf = lookup_tree.leaf("weights")?;
                weights_leaf.validate_shape(&[vocab_size as usize, model_dim as usize], data_type)?;
                let weights = weights_leaf.read_allocation()?;

                let lookup = <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel::new(context, data_type)
                    .map_err(EmbeddingError::BackendError)?;
                let readout = RefCell::new(<B::Kernels as ManualKernels>::MatmulKernel::new(context, data_type)?);

                EmbeddingTying::Tied {
                    ty: TiedEmbeddingType::FullPrecision {
                        weights,
                        lookup,
                        readout,
                    },
                }
            },
            EmbeddingConfig::Untied {
                common: _,
                precision,
            } => {
                let data_type = (*precision).into();

                let input_weights_leaf = lookup_tree.leaf(untied_keys.lookup_weights)?;
                input_weights_leaf.validate_shape(&[vocab_size as usize, model_dim as usize], data_type)?;
                let output_weights_leaf = readout_tree.leaf(untied_keys.readout_weights)?;
                output_weights_leaf.validate_shape(&[vocab_size as usize, model_dim as usize], data_type)?;

                let input_weights = input_weights_leaf.read_allocation()?;
                let output_weights = output_weights_leaf.read_allocation()?;

                let lookup = <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel::new(context, data_type)
                    .map_err(EmbeddingError::BackendError)?;
                let readout = RefCell::new(<B::Kernels as ManualKernels>::MatmulKernel::new(context, data_type)?);

                EmbeddingTying::Untied {
                    input_ty: UntiedEmbeddingLookupType::FullPrecision {
                        weights: input_weights,
                        lookup,
                    },
                    output_ty: UntiedEmbeddingReadoutType::FullPrecision {
                        weights: output_weights,
                        readout,
                    },
                }
            },
            EmbeddingConfig::ScaleBiasQuantizedTied {
                common: _,
                group_size,
                embedding_quantization_mode,
                activation_quantization_mode,
                activation_precision,
            } => {
                let data_type = (*activation_precision).into();

                let packing_divisor = embedding_quantization_mode.packing_divisor();
                let storage_data_type = embedding_quantization_mode.storage_type();

                let num_groups = (model_dim as usize).div_ceil(*group_size);

                let weights_leaf = lookup_tree.leaf("weights")?;
                weights_leaf
                    .validate_shape(&[vocab_size as usize, model_dim as usize / packing_divisor], storage_data_type)?;
                let scales_leaf = lookup_tree.leaf("scales")?;
                scales_leaf.validate_shape(&[vocab_size as usize, num_groups], data_type)?;
                let biases_leaf = lookup_tree.leaf("biases")?;
                biases_leaf.validate_shape(&[vocab_size as usize, num_groups], data_type)?;

                let weights = weights_leaf.read_allocation()?;
                let scales = scales_leaf.read_allocation()?;
                let biases = biases_leaf.read_allocation()?;

                let lookup = <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel::new(
                    context,
                    data_type,
                    *group_size as u32,
                    (*embedding_quantization_mode).into(),
                )
                .map_err(EmbeddingError::BackendError)?;
                let readout_config = QuantizedMatmulConfiguration {
                    data_type,
                    group_size: *group_size,
                    input_dim: model_dim as usize,
                    output_dim: vocab_size as usize,
                    mode: *embedding_quantization_mode,
                    quantization_method: QuantizationMethod::ScaleBias,
                    use_hadamard: false,
                };
                let readout = RefCell::new(
                    <B::Kernels as ManualKernels>::MatmulKernel::new(context, data_type)?,
                );

                if let Some(activation_quantization_mode) = activation_quantization_mode {
                    return Err(EmbeddingError::UnsupportedConfiguration(format!(
                        "activation_quantization_mode={activation_quantization_mode:?}"
                    )));
                }

                EmbeddingTying::Tied {
                    ty: TiedEmbeddingType::Quantized {
                        weights,
                        scales,
                        biases,
                        lookup,
                        readout,
                        readout_config,
                    },
                }
            },
            EmbeddingConfig::ScaleBiasQuantizedUntied {
                common: _,
                group_size,
                embedding_quantization_mode,
                activation_quantization_mode,
                activation_precision,
            } => {
                let data_type = (*activation_precision).into();

                let packing_divisor = embedding_quantization_mode.packing_divisor();
                let storage_data_type = embedding_quantization_mode.storage_type();

                let num_groups = (model_dim as usize).div_ceil(*group_size);

                let input_weights_leaf = lookup_tree.leaf(untied_keys.lookup_weights)?;
                input_weights_leaf
                    .validate_shape(&[vocab_size as usize, model_dim as usize / packing_divisor], storage_data_type)?;
                let input_scales_leaf = lookup_tree.leaf(untied_keys.lookup_scales)?;
                input_scales_leaf.validate_shape(&[vocab_size as usize, num_groups], data_type)?;
                let input_biases_leaf = lookup_tree.leaf(untied_keys.lookup_biases)?;
                input_biases_leaf.validate_shape(&[vocab_size as usize, num_groups], data_type)?;

                let output_weights_leaf = readout_tree.leaf(untied_keys.readout_weights)?;
                output_weights_leaf
                    .validate_shape(&[vocab_size as usize, model_dim as usize / packing_divisor], storage_data_type)?;
                let output_scales_leaf = readout_tree.leaf(untied_keys.readout_scales)?;
                output_scales_leaf.validate_shape(&[vocab_size as usize, num_groups], data_type)?;
                let output_biases_leaf = readout_tree.leaf(untied_keys.readout_biases)?;
                output_biases_leaf.validate_shape(&[vocab_size as usize, num_groups], data_type)?;

                let input_weights = input_weights_leaf.read_allocation()?;
                let input_scales = input_scales_leaf.read_allocation()?;
                let input_biases = input_biases_leaf.read_allocation()?;

                let output_weights = output_weights_leaf.read_allocation()?;
                let output_scales = output_scales_leaf.read_allocation()?;
                let output_biases = output_biases_leaf.read_allocation()?;

                let lookup = <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel::new(
                    context,
                    data_type,
                    *group_size as u32,
                    (*embedding_quantization_mode).into(),
                )
                .map_err(EmbeddingError::BackendError)?;
                let readout_config = QuantizedMatmulConfiguration {
                    data_type,
                    group_size: *group_size,
                    input_dim: model_dim as usize,
                    output_dim: vocab_size as usize,
                    mode: *embedding_quantization_mode,
                    quantization_method: QuantizationMethod::ScaleBias,
                    use_hadamard: false,
                };
                let readout = RefCell::new(
                    <B::Kernels as ManualKernels>::MatmulKernel::new(context, data_type)?,
                );

                if let Some(activation_quantization_mode) = activation_quantization_mode {
                    return Err(EmbeddingError::UnsupportedConfiguration(format!(
                        "activation_quantization_mode={activation_quantization_mode:?}"
                    )));
                }

                EmbeddingTying::Untied {
                    input_ty: UntiedEmbeddingLookupType::Quantized {
                        weights: input_weights,
                        scales: input_scales,
                        biases: input_biases,
                        lookup,
                    },
                    output_ty: UntiedEmbeddingReadoutType::Quantized {
                        weights: output_weights,
                        scales: output_scales,
                        biases: output_biases,
                        readout,
                        readout_config,
                    },
                }
            },
            EmbeddingConfig::ScaleBiasSemiQuantizedUntied {
                common: _,
                group_size,
                embedding_quantization_mode,
                activation_quantization_mode,
                activation_precision,
            } => {
                let data_type = (*activation_precision).into();

                let packing_divisor = embedding_quantization_mode.packing_divisor();
                let storage_data_type = embedding_quantization_mode.storage_type();

                let num_groups = (model_dim as usize).div_ceil(*group_size);

                let input_weights_leaf = lookup_tree.leaf(untied_keys.lookup_weights)?;
                input_weights_leaf.validate_shape(&[vocab_size as usize, model_dim as usize], data_type)?;

                let output_weights_leaf = readout_tree.leaf(untied_keys.readout_weights)?;
                output_weights_leaf
                    .validate_shape(&[vocab_size as usize, model_dim as usize / packing_divisor], storage_data_type)?;
                let output_scales_leaf = readout_tree.leaf(untied_keys.readout_scales)?;
                output_scales_leaf.validate_shape(&[vocab_size as usize, num_groups], data_type)?;
                let output_biases_leaf = readout_tree.leaf(untied_keys.readout_biases)?;
                output_biases_leaf.validate_shape(&[vocab_size as usize, num_groups], data_type)?;

                let input_weights = input_weights_leaf.read_allocation()?;

                let output_weights = output_weights_leaf.read_allocation()?;
                let output_scales = output_scales_leaf.read_allocation()?;
                let output_biases = output_biases_leaf.read_allocation()?;

                let lookup = <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel::new(context, data_type)
                    .map_err(EmbeddingError::BackendError)?;
                let readout_config = QuantizedMatmulConfiguration {
                    data_type,
                    group_size: *group_size,
                    input_dim: model_dim as usize,
                    output_dim: vocab_size as usize,
                    mode: *embedding_quantization_mode,
                    quantization_method: QuantizationMethod::ScaleBias,
                    use_hadamard: false,
                };
                let readout = RefCell::new(
                    <B::Kernels as ManualKernels>::MatmulKernel::new(context, data_type)?,
                );

                if let Some(activation_quantization_mode) = activation_quantization_mode {
                    return Err(EmbeddingError::UnsupportedConfiguration(format!(
                        "activation_quantization_mode={activation_quantization_mode:?}"
                    )));
                }

                EmbeddingTying::Untied {
                    input_ty: UntiedEmbeddingLookupType::FullPrecision {
                        weights: input_weights,
                        lookup,
                    },
                    output_ty: UntiedEmbeddingReadoutType::Quantized {
                        weights: output_weights,
                        scales: output_scales,
                        biases: output_biases,
                        readout,
                        readout_config,
                    },
                }
            },
        };

        let input_scale = common.input_scale.unwrap_or(1.0);
        let activation_data_type: DataType = config.activation_precision().into();

        if let Some(logit_soft_cap) = common.logit_soft_cap {
            return Err(EmbeddingError::UnsupportedConfiguration(format!("logit_soft_cap={logit_soft_cap:?}")));
        }

        Ok(Self {
            tying,
            input_scale,
            activation_data_type,
            vocab_size,
            model_dim,
        })
    }

    pub fn encode_lookup(
        &self,
        token_ids: &Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, EmbeddingError<B>> {
        let mut output = encoder
            .allocate_scratch(crate::array::size_for_shape(
                &[batch_dim, self.model_dim as usize],
                self.activation_data_type,
            ))
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
                        biases,
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
                        biases,
                        lookup,
                    },
                output_ty: _,
            } => {
                lookup.encode(
                    token_ids,
                    weights,
                    scales,
                    biases,
                    &mut output,
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
            .allocate_scratch(crate::array::size_for_shape(
                &[batch_dim, self.vocab_size as usize],
                self.activation_data_type,
            ))
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
                let input_dim = self.model_dim as usize;
                let output_dim = self.vocab_size as usize;
                readout.borrow_mut().encode(
                    MatmulArguments {
                        a: input_allocation,
                        a_offset: 0,
                        b: weights,
                        b_offset: 0,
                        b_leading_dimension: None,
                        b_transpose: true,
                        ab_scale: 1.0,
                        c: MatmulArgumentC::None,
                        d: &mut output_allocation,
                        batch_dim: batch_dim as u32,
                        input_dim: input_dim as u32,
                        output_dim: output_dim as u32,
                    },
                    encoder,
                );
            },
            EmbeddingTying::Tied {
                ty:
                    TiedEmbeddingType::Quantized {
                        weights,
                        scales,
                        biases,
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
                        biases,
                        readout,
                        readout_config,
                    },
            } => {
                readout
                    .borrow_mut()
                    .encode_quantized(
                        QuantizedMatmulArguments {
                            a: input_allocation,
                            a_offset: 0,
                            b: weights,
                            scales,
                            zero_points_or_biases: biases,
                            output: &mut output_allocation,
                            hadamard_factors: None,
                            batch_dim,
                        },
                        readout_config,
                        encoder,
                    )
                    .expect("encode_quantized failed");
            },
        };

        Ok(output_allocation)
    }
}
