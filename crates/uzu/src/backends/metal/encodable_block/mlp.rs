//! MLP block encodable.

use metal::CommandBufferRef;

use super::{EncodableBlock, EncodingParameters, QuantizedLinear};
use crate::backends::metal::{
    forward_pass::{ArrayId, ForwardPassState},
    kernel::mlp::MlpGateActMulEncodable,
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
        // Up
        self.up.encode(state, command_buffer, params);
        // Gate act+mul (fused_up -> hidden)
        let arrays = state.arrays(&[ArrayId::MlpFusedUp, ArrayId::MlpHidden]);
        let mut fused = arrays[0].borrow_mut();
        let mut hidden = arrays[1].borrow_mut();
        let active_suffix_length = state.active_suffix_length() as i32;
        let encoder = command_buffer.new_compute_command_encoder();
        let fused_buf = unsafe { fused.mtl_buffer() };
        let hidden_buf = unsafe { hidden.mtl_buffer() };
        self.gate
            .encode(encoder, fused_buf, hidden_buf, active_suffix_length)
            .expect("Failed to encode MLP activation/mul kernel");
        encoder.end_encoding();
        drop(fused);
        drop(hidden);
        // Down
        self.down.encode(state, command_buffer, params);

        if params.wait_until_completed {
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
    }
}
