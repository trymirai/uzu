//! Quantized embedding lookup encodable.

use thiserror::Error;

use super::{EncodableBlock, EncodingParameters};
use crate::{
    DataType,
    backends::common::{
        Backend, Context, NativeBuffer,
        kernel::{Kernels, QuantizedEmbeddingLookupKernel},
    },
    config::QuantizationMode,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum QuantizedEmbeddingLookupError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(ParameterLoaderError),
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Expected packed weights of type {expected:?}, got {got:?}")]
    InvalidWeightsDataType {
        expected: DataType,
        got: DataType,
    },
    #[error(
        "Embedding lookup weights shape mismatch: got {got:?}, expected [{expected_vocab_size}, {expected_packed_dim}]"
    )]
    InvalidWeightsShape {
        got: Box<[usize]>,
        expected_vocab_size: usize,
        expected_packed_dim: usize,
    },
    #[error(
        "Embedding lookup scales shape mismatch: got {got:?}, expected [{expected_vocab_size}, {expected_num_groups}]"
    )]
    InvalidScalesShape {
        got: Box<[usize]>,
        expected_vocab_size: usize,
        expected_num_groups: usize,
    },
    #[error("Scales dtype mismatch: got {got:?}, expected {expected:?}")]
    InvalidScalesDataType {
        expected: DataType,
        got: DataType,
    },
    #[error(
        "Embedding lookup biases shape mismatch: got {got:?}, expected [{expected_vocab_size}, {expected_num_groups}]"
    )]
    InvalidBiasesShape {
        got: Box<[usize]>,
        expected_vocab_size: usize,
        expected_num_groups: usize,
    },
    #[error("Biases dtype mismatch: got {got:?}, expected {expected:?}")]
    InvalidBiasesDataType {
        expected: DataType,
        got: DataType,
    },
}

pub struct QuantizedEmbeddingLookup<B: Backend> {
    kernel: <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel,
    weights_buffer: B::NativeBuffer,
    scales_buffer: B::NativeBuffer,
    biases_buffer: B::NativeBuffer,
    mode: QuantizationMode,
    input_scale: f32,
    vocab_size: u32,
    model_dim: u32,
    group_size: u32,
}

impl<B: Backend> QuantizedEmbeddingLookup<B> {
    pub fn new_tied(
        context: &B::Context,
        data_type: DataType,
        vocab_size: usize,
        model_dim: usize,
        group_size: usize,
        mode: QuantizationMode,
        input_scale: f32,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, QuantizedEmbeddingLookupError<B>> {
        Self::new_with_names(
            context,
            data_type,
            vocab_size,
            model_dim,
            group_size,
            mode,
            input_scale,
            "weights",
            "scales",
            "biases",
            parameter_tree,
        )
    }

    pub fn new_untied_input(
        context: &B::Context,
        data_type: DataType,
        vocab_size: usize,
        model_dim: usize,
        group_size: usize,
        mode: QuantizationMode,
        input_scale: f32,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, QuantizedEmbeddingLookupError<B>> {
        Self::new_with_names(
            context,
            data_type,
            vocab_size,
            model_dim,
            group_size,
            mode,
            input_scale,
            "input_weights",
            "input_scales",
            "input_biases",
            parameter_tree,
        )
    }

    fn new_with_names(
        context: &B::Context,
        data_type: DataType,
        vocab_size: usize,
        model_dim: usize,
        group_size: usize,
        mode: QuantizationMode,
        input_scale: f32,
        weights_name: &str,
        scales_name: &str,
        biases_name: &str,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, QuantizedEmbeddingLookupError<B>> {
        let packing_divisor = mode.packing_divisor();

        let kernel = <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel::new(context, data_type)
            .map_err(QuantizedEmbeddingLookupError::BackendError)?;

        let weights = parameter_tree.leaf(weights_name).map_err(QuantizedEmbeddingLookupError::ParameterError)?;

        if weights.data_type() != mode.storage_type() {
            return Err(QuantizedEmbeddingLookupError::InvalidWeightsDataType {
                expected: mode.storage_type(),
                got: weights.data_type(),
            });
        }

        let scales = parameter_tree.leaf(scales_name).map_err(QuantizedEmbeddingLookupError::ParameterError)?;

        let num_groups = (model_dim + group_size - 1) / group_size;
        if weights.shape() != [vocab_size, model_dim / packing_divisor] {
            return Err(QuantizedEmbeddingLookupError::InvalidWeightsShape {
                got: weights.shape().to_vec().into_boxed_slice(),
                expected_vocab_size: vocab_size,
                expected_packed_dim: model_dim / packing_divisor,
            });
        }

        if scales.shape() != [vocab_size, num_groups] {
            return Err(QuantizedEmbeddingLookupError::InvalidScalesShape {
                got: scales.shape().to_vec().into_boxed_slice(),
                expected_vocab_size: vocab_size,
                expected_num_groups: num_groups,
            });
        }

        if scales.data_type() != data_type {
            return Err(QuantizedEmbeddingLookupError::InvalidScalesDataType {
                expected: data_type,
                got: scales.data_type(),
            });
        }

        let biases_buffer = match parameter_tree.leaf(biases_name) {
            Ok(biases) => {
                if biases.shape() != [vocab_size, num_groups] {
                    return Err(QuantizedEmbeddingLookupError::InvalidBiasesShape {
                        got: biases.shape().to_vec().into_boxed_slice(),
                        expected_vocab_size: vocab_size,
                        expected_num_groups: num_groups,
                    });
                }

                if biases.data_type() != data_type {
                    return Err(QuantizedEmbeddingLookupError::InvalidBiasesDataType {
                        expected: data_type,
                        got: biases.data_type(),
                    });
                }

                biases.buffer().clone()
            },
            Err(_) => {
                let element_size = match data_type {
                    DataType::F16 | DataType::BF16 => 2,
                    DataType::F32 => 4,
                    other => {
                        return Err(QuantizedEmbeddingLookupError::UnsupportedDataType(other));
                    },
                };

                let size_bytes = vocab_size * num_groups * element_size;
                let buffer = context.create_buffer(size_bytes).map_err(QuantizedEmbeddingLookupError::BackendError)?;

                unsafe {
                    std::ptr::write_bytes(buffer.cpu_ptr().as_ptr().cast::<u8>(), 0, size_bytes);
                }

                buffer
            },
        };

        Ok(Self {
            kernel,
            weights_buffer: weights.buffer().clone(),
            scales_buffer: scales.buffer().clone(),
            biases_buffer,
            mode,
            input_scale,
            vocab_size: vocab_size as u32,
            model_dim: model_dim as u32,
            group_size: group_size as u32,
        })
    }
}

impl<B: Backend> EncodableBlock<B> for QuantizedEmbeddingLookup<B> {
    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<B>,
        _parameters: &EncodingParameters<B>,
        encoder: &B::ComputeEncoder,
    ) {
        let arrays = state.arrays(&[ArrayId::TokenIds, ArrayId::Main]);
        let batch_size = state.active_suffix_length();
        let token_ids_array = arrays[0].borrow_mut();
        let output_array = arrays[1].borrow_mut();

        let quant_mode = match self.mode {
            QuantizationMode::UInt4 => 0,
            QuantizationMode::Int8 => 1,
            QuantizationMode::UInt8 => 2,
        };

        self.kernel.encode(
            token_ids_array.buffer(),
            &self.weights_buffer,
            &self.scales_buffer,
            &self.biases_buffer,
            output_array.buffer(),
            batch_size as u32,
            self.vocab_size,
            self.model_dim,
            self.group_size,
            self.input_scale,
            quant_mode,
            encoder,
        );
    }
}
