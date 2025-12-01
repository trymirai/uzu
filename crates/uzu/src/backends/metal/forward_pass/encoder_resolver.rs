use metal::{ComputeCommandEncoder, MTLCompareFunction};
use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{
    super::metal_extensions::ComputeEncoderRawConditional,
    ForwardPassState,
    encodable_with_state::{EncodableWithState, EncodingParameters},
};

pub struct EncoderResolver<'a> {
    command_buffer: &'a MPSCommandBuffer,
    encoder: Option<ComputeCommandEncoder>,
    /// Precondition buffer and offset for conditional execution.
    /// If set, work is skipped when precondition[offset] != 0.
    precondition: Option<(metal::Buffer, usize)>,
}

impl<'a> EncoderResolver<'a> {
    pub fn new(
        command_buffer: &'a MPSCommandBuffer,
        precondition: Option<(metal::Buffer, usize)>,
    ) -> Self {
        Self {
            command_buffer,
            encoder: None,
            precondition,
        }
    }

    pub fn encode(
        &mut self,
        block: &dyn EncodableWithState,
        state: &mut ForwardPassState,
        parameters: &EncodingParameters,
    ) {
        if block.supports_shared_encoder() {
            if self.encoder.is_none() {
                let encoder = self
                    .command_buffer
                    .root_command_buffer()
                    .new_compute_command_encoder()
                    .to_owned();

                if let Some((buffer, offset)) = &self.precondition {
                    unsafe {
                        encoder.encode_start_if(
                            buffer,
                            *offset * std::mem::size_of::<u32>(),
                            MTLCompareFunction::Equal,
                            0,
                        );
                    }
                }

                self.encoder = Some(encoder);
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
            if self.precondition.is_some() {
                unsafe {
                    encoder.encode_end_if();
                }
            }
            encoder.end_encoding();
        }
    }

    pub fn finish(mut self) {
        self.end_current_encoder();
    }
}
