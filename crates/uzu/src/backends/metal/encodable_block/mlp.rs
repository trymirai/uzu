//! MLP block encodable.

use metal::{CommandBufferRef, ComputeCommandEncoderRef};

use super::{EncodableBlock, EncodingParameters};
use crate::{
    Array,
    backends::metal::{
    forward_pass::{ArrayId, ForwardPassState},
    kernel::mlp::MlpGateActMulEncodable,
    },
};

pub struct MlpBlock {
    up: Box<dyn EncodableBlock>,
    gate: MlpGateActMulEncodable,
    down: Box<dyn EncodableBlock>,
}

impl MlpBlock {
    pub fn new(
        up: Box<dyn EncodableBlock>,
        gate: MlpGateActMulEncodable,
        down: Box<dyn EncodableBlock>,
    ) -> Self {
        Self {
            up,
            gate,
            down,
        }
    }
}

impl EncodableBlock for MlpBlock {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &CommandBufferRef,
        params: &EncodingParameters,
    ) {
        if self.supports_shared_encoder() {
            let encoder = command_buffer.new_compute_command_encoder();
            self.encode_with_shared_encoder(state, &encoder, params);
            encoder.end_encoding();
        } else {
            // Up
            self.up.encode(state, command_buffer, params);

            // Gate act+mul (fused_up -> hidden)
            {
                let arrays =
                    state.arrays(&[ArrayId::MlpFusedUp, ArrayId::MlpHidden]);
                let mut fused = arrays[0].borrow_mut();
                let mut hidden = arrays[1].borrow_mut();
                let m = fused.shape()[0] as i32;
                let fused_buf = unsafe { fused.mtl_buffer() };
                let hidden_buf = unsafe { hidden.mtl_buffer() };

                let encoder = command_buffer.new_compute_command_encoder();
                self.gate
                    .encode(&encoder, fused_buf, hidden_buf, m)
                    .expect("Failed to encode MLP activation/mul kernel");
                encoder.end_encoding();
            }

            // Down
            self.down.encode(state, command_buffer, params);
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
        state: &mut ForwardPassState,
        encoder: &ComputeCommandEncoderRef,
        params: &EncodingParameters,
    ) {
        // Up
        self.up.encode_with_shared_encoder(state, encoder, params);

        // Gate act+mul (fused_up -> hidden)
        let arrays = state.arrays(&[ArrayId::MlpFusedUp, ArrayId::MlpHidden]);
        let mut fused = arrays[0].borrow_mut();
        let mut hidden = arrays[1].borrow_mut();
        let m = fused.shape()[0] as i32;
        let fused_buf = unsafe { fused.mtl_buffer() };
        let hidden_buf = unsafe { hidden.mtl_buffer() };
        self.gate
            .encode(encoder, fused_buf, hidden_buf, m)
            .expect("Failed to encode MLP activation/mul kernel");
        drop(fused);
        drop(hidden);

        // Down
        self.down.encode_with_shared_encoder(state, encoder, params);
    }
}
