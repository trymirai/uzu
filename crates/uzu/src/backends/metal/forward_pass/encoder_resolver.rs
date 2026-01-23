use super::ForwardPassState;
use crate::backends::metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, ProtocolObject, Retained,
    encodable_block::{EncodableBlock, EncodingParameters},
};

pub struct EncoderResolver<'a> {
    command_buffer: &'a ProtocolObject<dyn MTLCommandBuffer>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
}

impl<'a> EncoderResolver<'a> {
    pub fn new(
        command_buffer: &'a ProtocolObject<dyn MTLCommandBuffer>
    ) -> Self {
        Self {
            command_buffer,
            encoder: None,
        }
    }

    pub fn encode(
        &mut self,
        block: &dyn EncodableBlock,
        state: &mut ForwardPassState,
        parameters: &EncodingParameters,
    ) {
        if block.supports_shared_encoder() {
            if self.encoder.is_none() {
                self.encoder = Some(
                    self.command_buffer
                        .new_compute_command_encoder()
                        .expect("Failed to create compute command encoder"),
                );
            }
            block.encode_with_shared_encoder(
                state,
                self.encoder.as_ref().unwrap(),
                parameters,
            );
        } else {
            self.end_current_encoder();
            block.encode(state, self.command_buffer, parameters);
        }
    }

    pub fn end_current_encoder(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            encoder.end_encoding();
        }
    }

    pub fn finish(mut self) {
        self.end_current_encoder();
    }
}
