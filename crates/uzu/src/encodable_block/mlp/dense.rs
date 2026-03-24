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
        let arrays = state.arrays(&[ArrayId::MlpFusedUp, ArrayId::MlpHidden]);
        let fused = &arrays[0];
        let hidden = &arrays[1];
        let m = fused.shape()[0] as i32;
        let fused_buf_rc = fused.buffer();
        let fused_buf_borrow = fused_buf_rc.borrow();
        let hidden_buf_rc = hidden.buffer();
        let mut hidden_buf_borrow = hidden_buf_rc.borrow_mut();
        self.gate
            .encode(encoder, fused_buf_borrow.deref(), hidden_buf_borrow.deref_mut(), m)
            .expect("Failed to encode MLP activation/mul kernel");
        drop(hidden_buf_borrow);
        drop(fused_buf_borrow);
        drop(arrays);

        // Down
        self.down.encode(state, encoder)?;
        Ok(())
    }
}
