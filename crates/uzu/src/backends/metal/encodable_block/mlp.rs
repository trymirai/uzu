//! MLP block encodable.

use metal::ComputeCommandEncoderRef;
use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{EncodableBlock, EncodingParameters, QuantizedLinear};
use crate::{
    Array,
    backends::metal::{
        forward_pass::{ArrayId, ForwardPassState},
        kernel::mlp::MlpGateActMulEncodable,
    },
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
        let root = command_buffer.root_command_buffer().to_owned();
        let encoder = root.new_compute_command_encoder();
        self.encode_with_shared_encoder(state, &encoder, params);
        encoder.end_encoding();

        if params.wait_until_completed {
            command_buffer.commit_and_continue();
            root.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        true
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
