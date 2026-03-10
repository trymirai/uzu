//! MLP block encodable.

use std::ops::{Deref, DerefMut};

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
        params: &EncodingParameters,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        // Up
        self.up.encode(state, params, command_buffer)?;

        // Gate act+mul (fused_up -> hidden)
        let arrays = state.arrays(&[ArrayId::MlpFusedUp, ArrayId::MlpHidden]);
        let fused = arrays[0].borrow_mut();
        let hidden = arrays[1].borrow_mut();
        let m = fused.shape()[0] as i32;
        let fused_buf_rc = fused.buffer();
        let fused_buf_borrow = fused_buf_rc.borrow();
        let hidden_buf_rc = hidden.buffer();
        let mut hidden_buf_borrow = hidden_buf_rc.borrow_mut();
        self.gate
            .encode(command_buffer, fused_buf_borrow.deref(), hidden_buf_borrow.deref_mut(), m)
            .expect("Failed to encode MLP activation/mul kernel");
        drop(hidden_buf_borrow);
        drop(fused_buf_borrow);
        drop(fused);
        drop(hidden);

        // Down
        self.down.encode(state, params, command_buffer)?;
        Ok(())
    }
}
