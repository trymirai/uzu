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
            FullPrecisionEmbeddingLookupKernel, QuantizedEmbeddingLookupKernel,
            matmul::{MatmulArguments, MatmulError, MatmulKernel, MatmulKernels},
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
        readout: RefCell<<B::Kernels as MatmulKernels>::MatmulKernel>,
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
        readout: RefCell<<B::Kernels as MatmulKernels>::MatmulKernel>,
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
        let common = config.common();

        let tying = match config {
            EmbeddingConfig::Tied {
                common: _,
                precision,
            } => {
                let data_type = (*precision).into();

                let weights_leaf = parameter_tree.leaf("weights")?;
                validate_tensor(&weights_leaf, [vocab_size as usize, model_dim as usize], data_type)?;

                let weights = weights_leaf.read_buffer()?;

                let lookup = <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel::new(context, data_type)
                    .map_err(EmbeddingError::BackendError)?;
                let readout = RefCell::new(<B::Kernels as MatmulKernels>::MatmulKernel::new(context, data_type)?);

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

                let input_weights_leaf = parameter_tree.leaf("input_weights")?;
                validate_tensor(&input_weights_leaf, [vocab_size as usize, model_dim as usize], data_type)?;
                let output_weights_leaf = parameter_tree.leaf("output_weights")?;
                validate_tensor(&output_weights_leaf, [vocab_size as usize, model_dim as usize], data_type)?;

                let input_weights = input_weights_leaf.read_buffer()?;
                let output_weights = output_weights_leaf.read_buffer()?;

                let lookup = <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel::new(context, data_type)
                    .map_err(EmbeddingError::BackendError)?;
                let readout = RefCell::new(<B::Kernels as MatmulKernels>::MatmulKernel::new(context, data_type)?);

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

                let weights_leaf = parameter_tree.leaf("weights")?;
                validate_tensor(
                    &weights_leaf,
                    [vocab_size as usize, model_dim as usize / packing_divisor],
                    storage_data_type,
                )?;
                let scales_leaf = parameter_tree.leaf("scales")?;
                validate_tensor(&scales_leaf, [vocab_size as usize, num_groups], data_type)?;
                let biases_leaf = parameter_tree.leaf("biases")?;
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
                        weights_transposed: true,
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

                let input_weights_leaf = parameter_tree.leaf("input_weights")?;
                validate_tensor(
                    &input_weights_leaf,
                    [vocab_size as usize, model_dim as usize / packing_divisor],
                    storage_data_type,
                )?;
                let input_scales_leaf = parameter_tree.leaf("input_scales")?;
                validate_tensor(&input_scales_leaf, [vocab_size as usize, num_groups], data_type)?;
                let input_biases_leaf = parameter_tree.leaf("input_biases")?;
                validate_tensor(&input_biases_leaf, [vocab_size as usize, num_groups], data_type)?;

                let output_weights_leaf = parameter_tree.leaf("output_weights")?;
                validate_tensor(
                    &output_weights_leaf,
                    [vocab_size as usize, model_dim as usize / packing_divisor],
                    storage_data_type,
                )?;
                let output_scales_leaf = parameter_tree.leaf("output_scales")?;
                validate_tensor(&output_scales_leaf, [vocab_size as usize, num_groups], data_type)?;
                let output_biases_leaf = parameter_tree.leaf("output_biases")?;
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
                        weights_transposed: true,
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

                let input_weights_leaf = parameter_tree.leaf("input_weights")?;
                validate_tensor(&input_weights_leaf, [vocab_size as usize, model_dim as usize], data_type)?;

                let output_weights_leaf = parameter_tree.leaf("output_weights")?;
                validate_tensor(
                    &output_weights_leaf,
                    [vocab_size as usize, model_dim as usize / packing_divisor],
                    storage_data_type,
                )?;
                let output_scales_leaf = parameter_tree.leaf("output_scales")?;
                validate_tensor(&output_scales_leaf, [vocab_size as usize, num_groups], data_type)?;
                let output_biases_leaf = parameter_tree.leaf("output_biases")?;
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
                        weights_transposed: true,
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
        let batch_size = state.active_suffix_length() as u32;

        let arrays = state.arrays(&[ArrayId::TokenIds, ArrayId::Main]);

        let token_ids_array = &arrays[0];
        let token_ids_buffer_rc = token_ids_array.buffer();
        let token_ids_buffer_borrow = token_ids_buffer_rc.borrow();
        let token_ids = token_ids_buffer_borrow.deref();

        let output_array = &arrays[1];
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
                batch_size,
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
                    batch_size,
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
        let batch_size = state.sampling_length();

        if batch_size == 0 {
            return Ok(());
        }

        let arrays = state.arrays(&[ArrayId::Main, ArrayId::Logits]);

        let input_array = &arrays[0];
        let input_buffer_rc = input_array.buffer();
        let input_buffer_borrow = input_buffer_rc.borrow();
        let input = input_buffer_borrow.deref();

        let output_array = &arrays[1];
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
                        d: output,
                        bias: None,
                        batch: batch_size as i32,
                        input_dim: input_dim as i32,
                        output_dim: output_dim as i32,
                        leading_dimension_a: input_dim as i32,
                        leading_dimension_b: input_dim as i32,
                        leading_dimension_d: output_dim as i32,
                        transpose_b: true,
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
                        batch: batch_size,
                        input_dim: self.model_dim as usize,
                        output_dim: self.vocab_size as usize,
                        quantization_type: QuantizedMatmulType::Mlx,
                    },
                )?;
            },
        };

        Ok(())
    }
}
