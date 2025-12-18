use std::rc::Rc;

use metal::Buffer as MTLBuffer;
use mpsgraph::CommandBuffer;

use super::QuantizedEmbeddingError;
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

use super::super::{EncodableBlock, EncodingParameters};

pub struct FullPrecisionEmbeddingLookup {
    kernel: FullPrecisionEmbeddingLookupKernel,
    weights_buffer: MTLBuffer,
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
    ) -> Result<Self, QuantizedEmbeddingError> {
        let kernel =
            FullPrecisionEmbeddingLookupKernel::new(mtl_context, data_type)?;

        let mut weights = match parameter_tree.leaf("weights") {
            Ok(weights) => weights,
            Err(_) => parameter_tree.leaf("input_weights").map_err(|e| {
                QuantizedEmbeddingError::MetalError(MTLError::Generic(format!(
                    "Failed to load weights: {:?}",
                    e
                )))
            })?,
        };

        if weights.shape() != [vocab_size, model_dim] {
            return Err(QuantizedEmbeddingError::MetalError(
                MTLError::Generic(format!(
                    "Embedding lookup weights shape mismatch: got {:?}, \
                     expected [{}, {}]",
                    weights.shape(),
                    vocab_size,
                    model_dim
                )),
            ));
        }

        if weights.data_type() != data_type {
            return Err(QuantizedEmbeddingError::MetalError(
                MTLError::Generic(format!(
                    "Weights dtype mismatch: got {:?}, expected {:?}",
                    weights.data_type(),
                    data_type
                )),
            ));
        }

        let weights_buffer = unsafe { weights.mtl_buffer().to_owned() };

        Ok(Self {
            kernel,
            weights_buffer,
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
        command_buffer: &CommandBuffer,
        parameters: &EncodingParameters,
    ) {
        let arrays = state.arrays(&[ArrayId::TokenIds, ArrayId::Main]);
        let batch_size = state.active_suffix_length();
        let mut token_ids_array_mut = arrays[0].borrow_mut();
        let mut output_array_mut = arrays[1].borrow_mut();

        let token_ids_buffer = unsafe { token_ids_array_mut.mtl_buffer() };
        let output_buffer = unsafe { output_array_mut.mtl_buffer() };

        let root_command_buffer = command_buffer.root_command_buffer();
        let encoder = root_command_buffer.new_compute_command_encoder();

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

        encoder.end_encoding();

        if parameters.wait_until_completed {
            let mtl_command_buffer =
                command_buffer.root_command_buffer().to_owned();
            command_buffer.commit_and_continue();
            mtl_command_buffer.wait_until_completed();
        }
    }
}

