//! MLP block encodable.

use super::{EncodableBlock, Metal};
use crate::{
    backends::metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, ProtocolObject, Retained,
        kernel::mlp::MlpGateActMulEncodable,
    },
    encodable_block::EncodingParameters,
    forward_pass::state::{ArrayId, ForwardPassState},
};

pub struct MlpBlock {
    up: Box<dyn EncodableBlock<Metal>>,
    gate: MlpGateActMulEncodable,
    down: Box<dyn EncodableBlock<Metal>>,
}

impl MlpBlock {
    pub fn new(
        up: Box<dyn EncodableBlock<Metal>>,
        gate: MlpGateActMulEncodable,
        down: Box<dyn EncodableBlock<Metal>>,
    ) -> Self {
        Self {
            up,
            gate,
            down,
        }
    }
}

impl EncodableBlock<Metal> for MlpBlock {
    fn encode(
        &self,
        state: &mut ForwardPassState<Metal>,
        params: &EncodingParameters<Metal>,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    ) {
        if self.supports_shared_encoder() {
            let encoder =
                command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
            self.encode_with_shared_encoder(state, params, &encoder);
            encoder.end_encoding();
        } else {
            // Up
            self.up.encode(state, params, command_buffer);

            // Gate act+mul (fused_up -> hidden)
            {
                let arrays = state.arrays(&[ArrayId::MlpFusedUp, ArrayId::MlpHidden]);
                let fused = arrays[0].borrow_mut();
                let hidden = arrays[1].borrow_mut();
                let m = fused.shape()[0] as i32;
                let fused_buf = fused.buffer();
                let hidden_buf = hidden.buffer();

                let encoder =
                    command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
                self.gate
                    .encode(&encoder, fused_buf, hidden_buf, m)
                    .expect("Failed to encode MLP activation/mul kernel");
                encoder.end_encoding();
            }

            // Down
            self.down.encode(state, params, command_buffer);
        }

        if params.wait_until_completed {
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        self.up.supports_shared_encoder() && self.down.supports_shared_encoder()
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<Metal>,
        params: &EncodingParameters<Metal>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    ) {
        // Up
        self.up.encode_with_shared_encoder(state, params, encoder);

        // Gate act+mul (fused_up -> hidden)
        let arrays = state.arrays(&[ArrayId::MlpFusedUp, ArrayId::MlpHidden]);
        let fused = arrays[0].borrow_mut();
        let hidden = arrays[1].borrow_mut();
        let m = fused.shape()[0] as i32;
        let fused_buf = fused.buffer();
        let hidden_buf = hidden.buffer();
        self.gate.encode(encoder, fused_buf, hidden_buf, m).expect("Failed to encode MLP activation/mul kernel");
        drop(fused);
        drop(hidden);

        // Down
        self.down.encode_with_shared_encoder(state, params, encoder);
    }
}
