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
    prefill_reduced_up: Option<Box<dyn Linear<B>>>,
    prefill_reduced_gate: Option<MlpGateActMulEncodable<B>>,
    prefill_reduced_down: Option<Box<dyn Linear<B>>>,
}

impl<B: Backend> DenseMlp<B> {
    pub fn new(
        up: Box<dyn Linear<B>>,
        gate: MlpGateActMulEncodable<B>,
        down: Box<dyn Linear<B>>,
        prefill_reduced_up: Option<Box<dyn Linear<B>>>,
        prefill_reduced_gate: Option<MlpGateActMulEncodable<B>>,
        prefill_reduced_down: Option<Box<dyn Linear<B>>>,
    ) -> Self {
        Self {
            up,
            gate,
            down,
            prefill_reduced_up,
            prefill_reduced_gate,
            prefill_reduced_down,
        }
    }

    fn uses_prefill_reduced(
        &self,
        state: &ForwardPassState<B>,
        parameters: &EncodingParameters,
    ) -> bool {
        (state.is_prefilling() || state.sampling_start() > 0)
            && parameters.projection_step.is_none()
            && self.prefill_reduced_up.is_some()
            && self.prefill_reduced_gate.is_some()
            && self.prefill_reduced_down.is_some()
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

    fn encode_with(
        up: &dyn Linear<B>,
        gate: &MlpGateActMulEncodable<B>,
        down: &dyn Linear<B>,
        state: &mut ForwardPassState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        up.encode(state, command_buffer)?;
        Self::encode_gate(gate, state, command_buffer);
        down.encode(state, command_buffer)?;
        Ok(())
    }
}

impl<B: Backend> Mlp<B> for DenseMlp<B> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        if self.uses_prefill_reduced(state, parameters) {
            return Self::encode_with(
                self.prefill_reduced_up.as_deref().expect("checked above"),
                self.prefill_reduced_gate.as_ref().expect("checked above"),
                self.prefill_reduced_down.as_deref().expect("checked above"),
                state,
                command_buffer,
            );
        }
        Self::encode_with(self.up.as_ref(), &self.gate, self.down.as_ref(), state, command_buffer)
    }
}
