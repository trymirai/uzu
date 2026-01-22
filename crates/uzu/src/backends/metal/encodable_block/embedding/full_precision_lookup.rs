use std::rc::Rc;

use crate::backends::metal::{ProtocolObject,
    Buffer, ComputeCommandEncoderRef, MTLCommandBuffer,
    MTLCommandEncoder,
};

use super::{
    super::{EncodableBlock, EncodingParameters},
    EmbeddingError,
};
use crate::{
    Array, DataType,
    backends::metal::{
        MTLContext, MTLError,
        forward_pass::{ArrayId, ForwardPassState},
        kernel::embedding::{
            FullPrecisionEmbeddingLookupArguments,
            FullPrecisionEmbeddingLookupKernel,
        },
    },
    parameters::ParameterTree,
};

pub struct FullPrecisionEmbeddingLookup {
    kernel: FullPrecisionEmbeddingLookupKernel,
    weights_buffer: Buffer,
    vocab_size: u32,
    model_dim: u32,
    input_scale: f32,
}

impl FullPrecisionEmbeddingLookup {
    pub fn new(
        mtl_context: &MTLContext,
        data_type: DataType,
        vocab_size: usize,
        model_dim: usize,
        input_scale: Option<f32>,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) -> Result<Self, EmbeddingError> {
        let kernel =
            FullPrecisionEmbeddingLookupKernel::new(mtl_context, data_type)?;

        let mut weights = match parameter_tree.leaf("weights") {
            Ok(weights) => weights,
            Err(_) => parameter_tree.leaf("input_weights").map_err(|e| {
                EmbeddingError::MetalError(MTLError::Generic(format!(
                    "Failed to load weights: {:?}",
                    e
                )))
            })?,
        };

        if weights.shape() != [vocab_size, model_dim] {
            return Err(EmbeddingError::MetalError(MTLError::Generic(
                format!(
                    "Embedding lookup weights shape mismatch: got {:?}, \
                     expected [{}, {}]",
                    weights.shape(),
                    vocab_size,
                    model_dim
                ),
            )));
        }

        if weights.data_type() != data_type {
            return Err(EmbeddingError::MetalError(MTLError::Generic(
                format!(
                    "Weights dtype mismatch: got {:?}, expected {:?}",
                    weights.data_type(),
                    data_type
                ),
            )));
        }

        let weights_buffer = unsafe { weights.mtl_buffer().to_owned() };

        Ok(Self {
            kernel,
            weights_buffer: weights_buffer.into(),
            vocab_size: vocab_size as u32,
            model_dim: model_dim as u32,
            input_scale: input_scale.unwrap_or(1.0),
        })
    }
}

impl EncodableBlock for FullPrecisionEmbeddingLookup {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        parameters: &EncodingParameters,
    ) {
        let encoder = command_buffer.new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        self.encode_with_shared_encoder(state, &encoder, parameters);
        encoder.end_encoding();

        if parameters.wait_until_completed {
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState,
        encoder: ComputeCommandEncoderRef<'_>,
        _parameters: &EncodingParameters,
    ) {
        let arrays = state.arrays(&[ArrayId::TokenIds, ArrayId::Main]);
        let batch_size = state.active_suffix_length();
        let mut token_ids_array_mut = arrays[0].borrow_mut();
        let mut output_array_mut = arrays[1].borrow_mut();

        let token_ids_buffer = unsafe { token_ids_array_mut.mtl_buffer() };
        let output_buffer = unsafe { output_array_mut.mtl_buffer() };

        let args = FullPrecisionEmbeddingLookupArguments {
            token_ids_buffer,
            weights_buffer: &self.weights_buffer,
            output_buffer,
            batch_size: batch_size as u32,
            vocab_size: self.vocab_size,
            model_dim: self.model_dim,
            input_scale: self.input_scale,
        };

        self.kernel
            .encode(encoder, args)
            .expect("Failed to encode full precision embedding lookup kernel");
    }
}
