//! MLP block encodable.

use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{EncodableBlock, EncodingParameters, QuantizedLinear};
use crate::backends::metal::{
    forward_pass::{ArrayId, ForwardPassState},
    kernel::mlp::MlpGateActMulEncodable,
};

pub struct MlpBlock {
    up: QuantizedLinear,
    gate: MlpGateActMulEncodable,
    down: QuantizedLinear,
}

impl MlpBlock {
    pub fn new(
        up: QuantizedLinear,
        gate: MlpGateActMulEncodable,
        down: QuantizedLinear,
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
        command_buffer: &MPSCommandBuffer,
        params: &EncodingParameters,
    ) {
        // Up
        self.up.encode(state, command_buffer, params);
        // Gate act+mul (fused_up -> hidden)
        let arrays = state.arrays(&[ArrayId::MlpFusedUp, ArrayId::MlpHidden]);
        let mut fused = arrays[0].borrow_mut();
        let mut hidden = arrays[1].borrow_mut();
        let active_suffix_length = state.active_suffix_length() as i32;
        let root = command_buffer.root_command_buffer();
        let encoder = root.new_compute_command_encoder();
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
            let mtl_command_buffer =
                command_buffer.root_command_buffer().to_owned();
            command_buffer.commit_and_continue();
            mtl_command_buffer.wait_until_completed();
        }
    }
}
