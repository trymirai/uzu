use std::{
    cell::RefCell,
    ops::{Deref, DerefMut},
};

use thiserror::Error;

use crate::{
    DataType,
    backends::common::{
        Backend, Encoder, Kernels,
        kernel::{
            FullPrecisionEmbeddingLookupKernel, ManualKernels, QuantizedEmbeddingLookupKernel,
            matmul::{MatmulArgumentC, MatmulArguments, MatmulError, MatmulKernel},
            quant_matmul::{
                QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulError,
                QuantizedMatmulKernelEncodable, QuantizedMatmulType,
            },
        },
    },
    config::EmbeddingConfig,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLeaf, ParameterLoaderError, ParameterTree},
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
    #[error("Invalid tensor: got {shape:?} @ {data_type:?}, expected {expected_shape:?} @ {expected_data_type:?}")]
    InvalidTensor {
        shape: Box<[usize]>,
        data_type: DataType,
        expected_shape: Box<[usize]>,
        expected_data_type: DataType,
    },
}

enum TiedEmbeddingType<B: Backend> {
    FullPrecision {
        weights: B::Buffer,
        lookup: <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel,
        readout: RefCell<<B::Kernels as ManualKernels>::MatmulKernel>,
    },
    Quantized {
        weights: B::Buffer,
        scales: B::Buffer,
        biases: B::Buffer,
        lookup: <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel,
        readout: QuantizedMatmulKernelEncodable<B>,
    },
}

enum UntiedEmbeddingLookupType<B: Backend> {
    FullPrecision {
        weights: B::Buffer,
        lookup: <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel,
    },
    Quantized {
        weights: B::Buffer,
        scales: B::Buffer,
        biases: B::Buffer,
        lookup: <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel,
    },
}

enum UntiedEmbeddingReadoutType<B: Backend> {
    FullPrecision {
        weights: B::Buffer,
        readout: RefCell<<B::Kernels as ManualKernels>::MatmulKernel>,
    },
    Quantized {
        weights: B::Buffer,
        scales: B::Buffer,
        biases: B::Buffer,
        readout: QuantizedMatmulKernelEncodable<B>,
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

fn validate_tensor<'file, 'context, 'leaf, B: Backend>(
    weights_leaf: &ParameterLeaf<'file, 'context, 'leaf, B::Context>,
    expected_shape: [usize; 2],
    expected_data_type: DataType,
) -> Result<(), EmbeddingError<B>> {
    let shape = weights_leaf.shape();
    let data_type = weights_leaf.data_type();

    if (shape, data_type) != (expected_shape.as_ref(), expected_data_type) {
        return Err(EmbeddingError::InvalidTensor {
            shape: shape.into(),
            data_type: weights_leaf.data_type(),
            expected_shape: expected_shape.into(),
            expected_data_type,
        });
    }

    Ok(())
}

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
                validate_tensor(&weights_leaf, [vocab_size as usize, model_dim as usize], data_type)?;
                let weights = weights_leaf.read_buffer()?;

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
                validate_tensor(&input_weights_leaf, [vocab_size as usize, model_dim as usize], data_type)?;
                let output_weights_leaf = readout_tree.leaf(untied_keys.readout_weights)?;
                validate_tensor(&output_weights_leaf, [vocab_size as usize, model_dim as usize], data_type)?;

                let input_weights = input_weights_leaf.read_buffer()?;
                let output_weights = output_weights_leaf.read_buffer()?;

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
            EmbeddingConfig::MLXQuantizedTied {
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
                validate_tensor(
                    &weights_leaf,
                    [vocab_size as usize, model_dim as usize / packing_divisor],
                    storage_data_type,
                )?;
                let scales_leaf = lookup_tree.leaf("scales")?;
                validate_tensor(&scales_leaf, [vocab_size as usize, num_groups], data_type)?;
                let biases_leaf = lookup_tree.leaf("biases")?;
                validate_tensor(&biases_leaf, [vocab_size as usize, num_groups], data_type)?;

                let weights = weights_leaf.read_buffer()?;
                let scales = scales_leaf.read_buffer()?;
                let biases = biases_leaf.read_buffer()?;

                let lookup = <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel::new(
                    context,
                    data_type,
                    *group_size as u32,
                    (*embedding_quantization_mode).into(),
                )
                .map_err(EmbeddingError::BackendError)?;
                let readout = QuantizedMatmulKernelEncodable::new(
                    context,
                    QuantizedMatmulConfiguration {
                        data_type,
                        group_size: *group_size,
                        input_dim: model_dim as usize,
                        output_dim: vocab_size as usize,
                        mode: *embedding_quantization_mode,
                        quantization_type: QuantizedMatmulType::Mlx,
                    },
                )?;

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
                    },
                }
            },
            EmbeddingConfig::MLXQuantizedUntied {
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
                validate_tensor(
                    &input_weights_leaf,
                    [vocab_size as usize, model_dim as usize / packing_divisor],
                    storage_data_type,
                )?;
                let input_scales_leaf = lookup_tree.leaf(untied_keys.lookup_scales)?;
                validate_tensor(&input_scales_leaf, [vocab_size as usize, num_groups], data_type)?;
                let input_biases_leaf = lookup_tree.leaf(untied_keys.lookup_biases)?;
                validate_tensor(&input_biases_leaf, [vocab_size as usize, num_groups], data_type)?;

                let output_weights_leaf = readout_tree.leaf(untied_keys.readout_weights)?;
                validate_tensor(
                    &output_weights_leaf,
                    [vocab_size as usize, model_dim as usize / packing_divisor],
                    storage_data_type,
                )?;
                let output_scales_leaf = readout_tree.leaf(untied_keys.readout_scales)?;
                validate_tensor(&output_scales_leaf, [vocab_size as usize, num_groups], data_type)?;
                let output_biases_leaf = readout_tree.leaf(untied_keys.readout_biases)?;
                validate_tensor(&output_biases_leaf, [vocab_size as usize, num_groups], data_type)?;

                let input_weights = input_weights_leaf.read_buffer()?;
                let input_scales = input_scales_leaf.read_buffer()?;
                let input_biases = input_biases_leaf.read_buffer()?;

                let output_weights = output_weights_leaf.read_buffer()?;
                let output_scales = output_scales_leaf.read_buffer()?;
                let output_biases = output_biases_leaf.read_buffer()?;

                let lookup = <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel::new(
                    context,
                    data_type,
                    *group_size as u32,
                    (*embedding_quantization_mode).into(),
                )
                .map_err(EmbeddingError::BackendError)?;
                let readout = QuantizedMatmulKernelEncodable::new(
                    context,
                    QuantizedMatmulConfiguration {
                        data_type,
                        group_size: *group_size,
                        input_dim: model_dim as usize,
                        output_dim: vocab_size as usize,
                        mode: *embedding_quantization_mode,
                        quantization_type: QuantizedMatmulType::Mlx,
                    },
                )?;

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
                    },
                }
            },
            EmbeddingConfig::MLXSemiQuantizedUntied {
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
                validate_tensor(&input_weights_leaf, [vocab_size as usize, model_dim as usize], data_type)?;

                let output_weights_leaf = readout_tree.leaf(untied_keys.readout_weights)?;
                validate_tensor(
                    &output_weights_leaf,
                    [vocab_size as usize, model_dim as usize / packing_divisor],
                    storage_data_type,
                )?;
                let output_scales_leaf = readout_tree.leaf(untied_keys.readout_scales)?;
                validate_tensor(&output_scales_leaf, [vocab_size as usize, num_groups], data_type)?;
                let output_biases_leaf = readout_tree.leaf(untied_keys.readout_biases)?;
                validate_tensor(&output_biases_leaf, [vocab_size as usize, num_groups], data_type)?;

                let input_weights = input_weights_leaf.read_buffer()?;

                let output_weights = output_weights_leaf.read_buffer()?;
                let output_scales = output_scales_leaf.read_buffer()?;
                let output_biases = output_biases_leaf.read_buffer()?;

                let lookup = <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel::new(context, data_type)
                    .map_err(EmbeddingError::BackendError)?;
                let readout = QuantizedMatmulKernelEncodable::new(
                    context,
                    QuantizedMatmulConfiguration {
                        data_type,
                        group_size: *group_size,
                        input_dim: model_dim as usize,
                        output_dim: vocab_size as usize,
                        mode: *embedding_quantization_mode,
                        quantization_type: QuantizedMatmulType::Mlx,
                    },
                )?;

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
                    },
                }
            },
        };

        let input_scale = common.input_scale.unwrap_or(1.0);

        if let Some(logit_soft_cap) = common.logit_soft_cap {
            return Err(EmbeddingError::UnsupportedConfiguration(format!("logit_soft_cap={logit_soft_cap:?}")));
        }

        Ok(Self {
            tying,
            input_scale,
            vocab_size,
            model_dim,
        })
    }

    pub fn encode_lookup(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), EmbeddingError<B>> {
        let batch_dim = state.active_row_count() as u32;

        let token_ids_array = state.array(ArrayId::TokenIds);
        let token_ids_buffer_rc = token_ids_array.buffer();
        let token_ids_buffer_borrow = token_ids_buffer_rc.borrow();
        let token_ids = token_ids_buffer_borrow.deref();

        let output_array = state.array(ArrayId::Main);
        let output_buffer_rc = output_array.buffer();
        let mut output_buffer_borrow = output_buffer_rc.borrow_mut();
        let output = output_buffer_borrow.deref_mut();

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
                output,
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
                    output,
                    batch_dim,
                    self.vocab_size,
                    self.model_dim,
                    self.input_scale,
                    encoder,
                );
            },
        };

        Ok(())
    }

    pub fn encode_readout(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), EmbeddingError<B>> {
        let batch_dim = state.sampling_length();

        if batch_dim == 0 {
            return Ok(());
        }

        let input_array = state.array(ArrayId::Main);
        let input_buffer_rc = input_array.buffer();
        let input_buffer_borrow = input_buffer_rc.borrow();
        let input = input_buffer_borrow.deref();

        let output_array = state.array(ArrayId::Logits);
        let output_buffer_rc = output_array.buffer();
        let mut output_buffer_borrow = output_buffer_rc.borrow_mut();
        let output = output_buffer_borrow.deref_mut();

        let input_offset = state.sampling_start() * self.model_dim as usize * input_array.data_type().size_in_bytes();

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
                    state.context(),
                    MatmulArguments {
                        a: input,
                        a_offset: input_offset as u64,
                        b: weights,
                        ab_scale: 1.0,
                        c: MatmulArgumentC::None,
                        d: output,
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
                    },
            } => {
                readout.encode(
                    encoder,
                    QuantizedMatmulArguments {
                        a_buffer: input,
                        a_offset: input_offset,
                        b_buffer: weights,
                        scales_buffer: scales,
                        zero_points_or_biases_buffer: biases,
                        output_buffer: output,
                        batch_dim,
                    },
                )?;
            },
        };

        Ok(())
    }
}
