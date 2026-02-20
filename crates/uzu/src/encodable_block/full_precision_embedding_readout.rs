use std::cell::RefCell;

use thiserror::Error;

use super::{EncodableBlock, EncodingParameters};
use crate::{
    DataType,
    backends::common::{
        Backend,
        kernel::matmul::{FullPrecisionMatmulArguments, FullPrecisionMatmulKernel, MatmulKernels},
    },
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum FullPrecisionEmbeddingReadoutError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(ParameterLoaderError),
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error(
        "Embedding readout weights shape mismatch: got {got:?}, expected [{expected_vocab_size}, {expected_model_dim}]"
    )]
    InvalidWeightsShape {
        got: Box<[usize]>,
        expected_vocab_size: usize,
        expected_model_dim: usize,
    },
    #[error("Weights dtype mismatch: got {got:?}, expected {expected:?}")]
    InvalidWeightsDataType {
        expected: DataType,
        got: DataType,
    },
}

pub struct FullPrecisionEmbeddingReadout<B: Backend>
where
    B::Kernels: MatmulKernels,
{
    kernel: RefCell<<B::Kernels as MatmulKernels>::FullPrecisionMatmulKernel>,
    weights_buffer: B::NativeBuffer,
    vocab_size: usize,
    model_dim: usize,
}

impl<B: Backend> FullPrecisionEmbeddingReadout<B>
where
    B::Kernels: MatmulKernels,
{
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        vocab_size: usize,
        model_dim: usize,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, FullPrecisionEmbeddingReadoutError<B>> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(FullPrecisionEmbeddingReadoutError::UnsupportedDataType(data_type));
        }

        let weights = match parameter_tree.leaf("weights") {
            Ok(weights) => weights,
            Err(_) => {
                parameter_tree.leaf("output_weights").map_err(FullPrecisionEmbeddingReadoutError::ParameterError)?
            },
        };

        let weights_shape = weights.shape().to_vec();
        if weights_shape != [vocab_size, model_dim] {
            return Err(FullPrecisionEmbeddingReadoutError::InvalidWeightsShape {
                got: weights_shape.into_boxed_slice(),
                expected_vocab_size: vocab_size,
                expected_model_dim: model_dim,
            });
        }

        if weights.data_type() != data_type {
            return Err(FullPrecisionEmbeddingReadoutError::InvalidWeightsDataType {
                expected: data_type,
                got: weights.data_type(),
            });
        }

        let kernel = <B::Kernels as MatmulKernels>::FullPrecisionMatmulKernel::new(context, data_type)
            .map_err(FullPrecisionEmbeddingReadoutError::BackendError)?;

        Ok(Self {
            kernel: RefCell::new(kernel),
            weights_buffer: weights.buffer().clone(),
            vocab_size,
            model_dim,
        })
    }
}

impl<B: Backend> EncodableBlock<B> for FullPrecisionEmbeddingReadout<B>
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

        self.kernel.borrow_mut().encode(
            state.context(),
            encoder,
            FullPrecisionMatmulArguments {
                a: input_array.buffer(),
                a_offset: sampling_start * self.model_dim * element_size,
                b: &self.weights_buffer,
                output: output_array.buffer(),
                bias: None,
                batch: batch_size,
                input_dim: self.model_dim,
                output_dim: self.vocab_size,
            },
        );
    }
}
