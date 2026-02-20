//! MLP block encodable.

use crate::{
    backends::common::{Backend, CommandBuffer, kernel::mlp_gate_act_mul::MlpGateActMulEncodable},
    encodable_block::{EncodableBlock, EncodingParameters},
    forward_pass::state::{ArrayId, ForwardPassState},
};

pub struct MlpBlock<B: Backend> {
    up: Box<dyn EncodableBlock<B>>,
    gate: MlpGateActMulEncodable<B>,
    down: Box<dyn EncodableBlock<B>>,
}

impl<B: Backend> MlpBlock<B> {
    pub fn new(
        up: Box<dyn EncodableBlock<B>>,
        gate: MlpGateActMulEncodable<B>,
        down: Box<dyn EncodableBlock<B>>,
    ) -> Self {
        Self {
            up,
            gate,
            down,
        }
    }
}

impl<B: Backend> EncodableBlock<B> for MlpBlock<B> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        params: &EncodingParameters<B>,
        command_buffer: &B::CommandBuffer,
    ) {
        if self.supports_shared_encoder() {
            command_buffer.with_compute_encoder(|encoder| self.encode_with_shared_encoder(state, params, encoder));
        } else {
            // Up
            self.up.encode(state, params, command_buffer);

            // Gate act+mul (fused_up -> hidden)
            command_buffer.with_compute_encoder(|encoder| {
                let arrays = state.arrays(&[ArrayId::MlpFusedUp, ArrayId::MlpHidden]);
                let fused = arrays[0].borrow_mut();
                let hidden = arrays[1].borrow_mut();
                let m = fused.shape()[0] as i32;
                let fused_buf = fused.buffer();
                let hidden_buf = hidden.buffer();

                self.gate
                    .encode(encoder, fused_buf, hidden_buf, m)
                    .expect("Failed to encode MLP activation/mul kernel");
            });

            // Down
            self.down.encode(state, params, command_buffer);
        }

        if params.wait_until_completed {
            command_buffer.submit();
            command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        self.up.supports_shared_encoder() && self.down.supports_shared_encoder()
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<B>,
        params: &EncodingParameters<B>,
        encoder: &B::ComputeEncoder,
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
