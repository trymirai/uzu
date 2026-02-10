use super::{
    super::{EncodableBlock, Metal},
    EmbeddingError,
};
use crate::{
    DataType,
    backends::{
        common::kernel::FullPrecisionEmbeddingLookupKernel,
        metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
            MTLComputeCommandEncoder, MTLContext, MTLError, ProtocolObject,
            Retained, kernel::dsl::FullPrecisionEmbeddingLookupMetalKernel,
        },
    },
    encodable_block::EncodingParameters,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::ParameterTree,
};

pub struct FullPrecisionEmbeddingLookup {
    kernel: FullPrecisionEmbeddingLookupMetalKernel,
    weights_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
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
        parameter_tree: &ParameterTree<MTLContext>,
    ) -> Result<Self, EmbeddingError> {
        let kernel = FullPrecisionEmbeddingLookupMetalKernel::new(
            mtl_context,
            data_type.into(),
        )?;

        let weights = match parameter_tree.leaf("weights") {
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

        let weights_buffer = weights.buffer().to_owned();

        Ok(Self {
            kernel,
            weights_buffer: weights_buffer.into(),
            vocab_size: vocab_size as u32,
            model_dim: model_dim as u32,
            input_scale: input_scale.unwrap_or(1.0),
        })
    }
}

impl EncodableBlock<Metal> for FullPrecisionEmbeddingLookup {
    fn encode(
        &self,
        state: &mut ForwardPassState<Metal>,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        parameters: &EncodingParameters<Metal>,
    ) {
        let encoder = command_buffer
            .new_compute_command_encoder()
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
        state: &mut ForwardPassState<Metal>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        _parameters: &EncodingParameters<Metal>,
    ) {
        let arrays = state.arrays(&[ArrayId::TokenIds, ArrayId::Main]);
        let batch_size = state.active_suffix_length();
        let token_ids_array_mut = arrays[0].borrow_mut();
        let output_array_mut = arrays[1].borrow_mut();

        let token_ids_buffer = token_ids_array_mut.buffer();
        let output_buffer = output_array_mut.buffer();

        self.kernel.encode(
            token_ids_buffer,
            &self.weights_buffer,
            output_buffer,
            batch_size as u32,
            self.vocab_size,
            self.model_dim,
            self.input_scale,
            encoder,
        )
    }
}
