//! Quantized embedding readout encodable.

use thiserror::Error;

use super::{EncodableBlock, EncodingParameters};
use crate::{
    DataType,
    backends::common::{
        Backend, Context, NativeBuffer,
        kernel::matmul::{
            MatmulKernels, QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulKernel,
            QuantizedMatmulType,
        },
    },
    config::QuantizationMode,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum QuantizedEmbeddingReadoutError<B: Backend>
where
    B::Kernels: MatmulKernels,
{
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(ParameterLoaderError),
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error(
        "Embedding readout weights shape mismatch: got {got:?}, expected [{expected_vocab_size}, {expected_packed_dim}]"
    )]
    InvalidWeightsShape {
        got: Box<[usize]>,
        expected_vocab_size: usize,
        expected_packed_dim: usize,
    },
    #[error(
        "Embedding readout scales shape mismatch: got {got:?}, expected [{expected_vocab_size}, {expected_num_groups}]"
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
        "Embedding readout deq_biases shape mismatch: got {got:?}, expected [{expected_vocab_size}, {expected_num_groups}]"
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

pub struct QuantizedEmbeddingReadout<B: Backend>
where
    B::Kernels: MatmulKernels,
{
    kernel: <B::Kernels as MatmulKernels>::QuantizedMatmulKernel,
    weights_buffer: B::NativeBuffer,
    scales_buffer: B::NativeBuffer,
    biases_buffer: B::NativeBuffer,
    vocab_size: usize,
    model_dim: usize,
}

impl<B: Backend> QuantizedEmbeddingReadout<B>
where
    B::Kernels: MatmulKernels,
{
    pub fn new_tied(
        context: &B::Context,
        data_type: DataType,
        vocab_size: usize,
        model_dim: usize,
        group_size: usize,
        mode: QuantizationMode,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, QuantizedEmbeddingReadoutError<B>> {
        Self::new_with_names(
            context,
            data_type,
            vocab_size,
            model_dim,
            group_size,
            mode,
            "weights",
            "scales",
            "biases",
            parameter_tree,
        )
    }

    pub fn new_untied_output(
        context: &B::Context,
        data_type: DataType,
        vocab_size: usize,
        model_dim: usize,
        group_size: usize,
        mode: QuantizationMode,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, QuantizedEmbeddingReadoutError<B>> {
        Self::new_with_names(
            context,
            data_type,
            vocab_size,
            model_dim,
            group_size,
            mode,
            "output_weights",
            "output_scales",
            "output_biases",
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
        weights_name: &str,
        scales_name: &str,
        biases_name: &str,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, QuantizedEmbeddingReadoutError<B>> {
        let weights = parameter_tree.leaf(weights_name).map_err(QuantizedEmbeddingReadoutError::ParameterError)?;
        let scales = parameter_tree.leaf(scales_name).map_err(QuantizedEmbeddingReadoutError::ParameterError)?;

        let num_groups = (model_dim + group_size - 1) / group_size;
        let packing_divisor = mode.packing_divisor();
        let weights_transposed = weights.shape()[0] == vocab_size;

        if weights.shape() != [vocab_size, model_dim / packing_divisor] {
            return Err(QuantizedEmbeddingReadoutError::InvalidWeightsShape {
                got: weights.shape().to_vec().into_boxed_slice(),
                expected_vocab_size: vocab_size,
                expected_packed_dim: model_dim / packing_divisor,
            });
        }

        if scales.shape() != [vocab_size, num_groups] {
            return Err(QuantizedEmbeddingReadoutError::InvalidScalesShape {
                got: scales.shape().to_vec().into_boxed_slice(),
                expected_vocab_size: vocab_size,
                expected_num_groups: num_groups,
            });
        }

        if scales.data_type() != data_type {
            return Err(QuantizedEmbeddingReadoutError::InvalidScalesDataType {
                expected: data_type,
                got: scales.data_type(),
            });
        }

        let biases_buffer = match parameter_tree.leaf(biases_name) {
            Ok(deq_biases) => {
                if deq_biases.shape() != [vocab_size, num_groups] {
                    return Err(QuantizedEmbeddingReadoutError::InvalidBiasesShape {
                        got: deq_biases.shape().to_vec().into_boxed_slice(),
                        expected_vocab_size: vocab_size,
                        expected_num_groups: num_groups,
                    });
                }
                if deq_biases.data_type() != data_type {
                    return Err(QuantizedEmbeddingReadoutError::InvalidBiasesDataType {
                        expected: data_type,
                        got: deq_biases.data_type(),
                    });
                }
                deq_biases.buffer().clone()
            },
            Err(_) => {
                let element_size = match data_type {
                    DataType::F16 | DataType::BF16 => 2,
                    DataType::F32 => 4,
                    other => {
                        return Err(QuantizedEmbeddingReadoutError::UnsupportedDataType(other));
                    },
                };
                let size_bytes = vocab_size * num_groups * element_size;
                let buffer = context.create_buffer(size_bytes).map_err(QuantizedEmbeddingReadoutError::BackendError)?;

                unsafe {
                    std::ptr::write_bytes(buffer.cpu_ptr().as_ptr().cast::<u8>(), 0, size_bytes);
                }

                buffer
            },
        };

        let kernel = <B::Kernels as MatmulKernels>::QuantizedMatmulKernel::new(
            context,
            QuantizedMatmulConfiguration {
                data_type,
                group_size,
                input_dim: model_dim,
                output_dim: vocab_size,
                mode,
                quantization_type: QuantizedMatmulType::Mlx,
                weights_transposed,
            },
        )
        .map_err(QuantizedEmbeddingReadoutError::BackendError)?;

        Ok(Self {
            kernel,
            weights_buffer: weights.buffer().clone(),
            scales_buffer: scales.buffer().clone(),
            biases_buffer,
            vocab_size,
            model_dim,
        })
    }
}

impl<B: Backend> EncodableBlock<B> for QuantizedEmbeddingReadout<B>
where
    B::Kernels: MatmulKernels,
{
    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<B>,
        _parameters: &EncodingParameters<B>,
        encoder: &B::ComputeEncoder,
    ) {
        let arrays = state.arrays(&[ArrayId::Main, ArrayId::Logits]);
        let batch_size = state.sampling_length();
        if batch_size == 0 {
            return;
        }

        let sampling_start = state.sampling_start();
        let input_array = arrays[0].borrow_mut();
        let output_array = arrays[1].borrow_mut();

        let element_size = input_array.data_type().size_in_bytes();
        let a_offset = sampling_start * self.model_dim * element_size;

        self.kernel.encode(
            encoder,
            QuantizedMatmulArguments {
                a_buffer: input_array.buffer(),
                a_offset,
                b_buffer: &self.weights_buffer,
                scales_buffer: &self.scales_buffer,
                zero_points_or_biases_buffer: &self.biases_buffer,
                output_buffer: output_array.buffer(),
                batch: batch_size,
                input_dim: self.model_dim,
                output_dim: self.vocab_size,
                quantization_type: QuantizedMatmulType::Mlx,
            },
        );
    }
}
