//! MLP block encodable.

use std::ops::{Deref, DerefMut};

use super::{super::linear::Linear, Mlp};
use crate::{
    backends::common::{Backend, CommandBuffer, kernel::mlp_gate_act_mul::MlpGateActMulEncodable},
    encodable_block::EncodingParameters,
    forward_pass::state::{ArrayId, ForwardPassState},
};

pub struct DenseMlp<B: Backend> {
    up: Box<dyn Linear<B>>,
    gate: MlpGateActMulEncodable<B>,
    down: Box<dyn Linear<B>>,
}

impl<B: Backend> DenseMlp<B> {
    pub(super) fn new(
        up: Box<dyn Linear<B>>,
        gate: MlpGateActMulEncodable<B>,
        down: Box<dyn Linear<B>>,
    ) -> Self {
        Self {
            up,
            gate,
            down,
        }
    }

    fn encode_gate(
        gate: &MlpGateActMulEncodable<B>,
        state: &mut ForwardPassState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        let arrays = state.arrays(&[ArrayId::MlpFusedUp, ArrayId::MlpHidden]);
        let fused = arrays[0].borrow_mut();
        let hidden = arrays[1].borrow_mut();
        let batch_size = fused.shape()[0] as i32;
        let fused_buffer = fused.buffer();
        let fused_buffer = fused_buffer.borrow();
        let hidden_buffer = hidden.buffer();
        let mut hidden_buffer = hidden_buffer.borrow_mut();
        gate.encode(command_buffer, fused_buffer.deref(), hidden_buffer.deref_mut(), batch_size)
            .expect("Failed to encode MLP activation/mul kernel");
    }
}

impl<B: Backend> Mlp<B> for DenseMlp<B> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        _parameters: &EncodingParameters,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        self.up.encode(state, command_buffer)?;
        Self::encode_gate(&self.gate, state, command_buffer);
        self.down.encode(state, command_buffer)
    }
}
