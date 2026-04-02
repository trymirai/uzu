//! MLP block encodable.

use std::ops::{Deref, DerefMut};

use super::{super::linear::Linear, Mlp};
use crate::{
    backends::common::{Backend, Encoder, kernel::mlp_gate_act_mul::MlpGateActMulEncodable},
    forward_pass::state::{ArrayId, ForwardPassState},
};

pub struct DenseMlp<B: Backend> {
    up: Box<dyn Linear<B>>,
    gate: MlpGateActMulEncodable<B>,
    down: Box<dyn Linear<B>>,
}

impl<B: Backend> DenseMlp<B> {
    pub fn new(
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
}

impl<B: Backend> Mlp<B> for DenseMlp<B> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        // Up
        self.up.encode(state, encoder)?;

        // Gate act+mul (fused_up -> hidden)
        {
            let fused = state.array(ArrayId::MlpFusedUp);
            let hidden = state.array(ArrayId::MlpHidden);
            let m = fused.shape()[0] as i32;
            self.gate.encode(encoder, fused.buffer().borrow().deref(), hidden.buffer().borrow_mut().deref_mut(), m)?;
        }

        // Down
        self.down.encode(state, encoder)?;
        Ok(())
    }
}
