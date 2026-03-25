//! MLP block encodable.

use std::ops::{Deref, DerefMut};

use super::{super::linear::Linear, Mlp};
use crate::{
    backends::common::{
        Backend, Encoder,
        kernel::mlp_gate_act_mul::{MlpGateActMulEncodable, MlpGateActMulHadamardEncodable},
    },
    forward_pass::state::{ArrayId, ForwardPassState},
};

pub struct DenseMlp<B: Backend> {
    up: Box<dyn Linear<B>>,
    gate: MlpGateActMulEncodable<B>,
    gate_hadamard: Option<MlpGateActMulHadamardEncodable<B>>,
    down: Box<dyn Linear<B>>,
}

impl<B: Backend> DenseMlp<B> {
    pub fn new(
        up: Box<dyn Linear<B>>,
        gate: MlpGateActMulEncodable<B>,
        gate_hadamard: Option<MlpGateActMulHadamardEncodable<B>>,
        down: Box<dyn Linear<B>>,
    ) -> Self {
        Self {
            up,
            gate,
            gate_hadamard,
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
        self.up.encode(state, encoder)?;

        let arrays = state.arrays(&[ArrayId::MlpFusedUp, ArrayId::MlpHidden]);
        let fused = &arrays[0];
        let hidden = &arrays[1];
        let m = fused.shape()[0] as i32;
        let fused_buf_rc = fused.buffer();
        let fused_buf_borrow = fused_buf_rc.borrow();
        let hidden_buf_rc = hidden.buffer();
        let mut hidden_buf_borrow = hidden_buf_rc.borrow_mut();

        if let Some(ref fused_kernel) = self.gate_hadamard {
            fused_kernel
                .encode(encoder, fused_buf_borrow.deref(), hidden_buf_borrow.deref_mut(), m)
                .expect("Failed to encode fused MLP activation/mul + Hadamard kernel");
        } else {
            self.gate
                .encode(encoder, fused_buf_borrow.deref(), hidden_buf_borrow.deref_mut(), m)
                .expect("Failed to encode MLP activation/mul kernel");
        }
        drop(hidden_buf_borrow);
        drop(fused_buf_borrow);
        drop(arrays);

        self.down.encode(state, encoder)?;
        Ok(())
    }
}
