//! Full precision embedding lookup encodable.

use thiserror::Error;

use super::{EncodableBlock, EncodingParameters};
use crate::{
    DataType,
    backends::common::{
        Backend,
        kernel::{FullPrecisionEmbeddingLookupKernel, Kernels},
    },
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum FullPrecisionEmbeddingLookupError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(ParameterLoaderError),
    #[error(
        "Embedding lookup weights shape mismatch: got {got:?}, expected [{expected_vocab_size}, {expected_model_dim}]"
    )]
    InvalidWeightsShape {
        got: Box<[usize]>,
        expected_vocab_size: usize,
        expected_model_dim: usize,
    },
    #[error("Weights dtype mismatch: got {got:?}, expected {expected:?}")]
    InvalidWeightsDataType {
        got: DataType,
        expected: DataType,
    },
}

pub struct FullPrecisionEmbeddingLookup<B: Backend> {
    kernel: <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel,
    weights_buffer: B::NativeBuffer,
    vocab_size: u32,
    model_dim: u32,
    input_scale: f32,
}

impl<B: Backend> FullPrecisionEmbeddingLookup<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        vocab_size: usize,
        model_dim: usize,
        input_scale: Option<f32>,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, FullPrecisionEmbeddingLookupError<B>> {
        let kernel = <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel::new(context, data_type)
            .map_err(FullPrecisionEmbeddingLookupError::BackendError)?;

        let weights = parameter_tree
            .leaf("weights")
            .or_else(|_| parameter_tree.leaf("input_weights"))
            .map_err(FullPrecisionEmbeddingLookupError::ParameterError)?;

        if weights.shape() != [vocab_size, model_dim] {
            return Err(FullPrecisionEmbeddingLookupError::InvalidWeightsShape {
                got: weights.shape().to_vec().into_boxed_slice(),
                expected_vocab_size: vocab_size,
                expected_model_dim: model_dim,
            });
        }

        if weights.data_type() != data_type {
            return Err(FullPrecisionEmbeddingLookupError::InvalidWeightsDataType {
                got: weights.data_type(),
                expected: data_type,
            });
        }

        Ok(Self {
            kernel,
            weights_buffer: weights.buffer().clone(),
            vocab_size: vocab_size as u32,
            model_dim: model_dim as u32,
            input_scale: input_scale.unwrap_or(1.0),
        })
    }
}

impl<B: Backend> EncodableBlock<B> for FullPrecisionEmbeddingLookup<B> {
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

        self.kernel.encode(
            token_ids_array.buffer(),
            &self.weights_buffer,
            output_array.buffer(),
            batch_size as u32,
            self.vocab_size,
            self.model_dim,
            self.input_scale,
            encoder,
        );
    }
}
