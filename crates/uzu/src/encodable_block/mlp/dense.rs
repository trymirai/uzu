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

        // Gate act+mul (fused_up -> hidden)
        {
            let fused = state.array(ArrayId::MlpFusedUp);
            let hidden = state.array(ArrayId::MlpHidden);
            let m = fused.shape()[0] as i32;
            if let Some(ref fused_kernel) = self.gate_hadamard {
                fused_kernel.encode(
                    encoder,
                    fused.buffer().borrow().deref(),
                    hidden.buffer().borrow_mut().deref_mut(),
                    m,
                )?;
            } else {
                self.gate.encode(
                    encoder,
                    fused.buffer().borrow().deref(),
                    hidden.buffer().borrow_mut().deref_mut(),
                    m,
                )?;
            }
        }

        self.down.encode(state, encoder)?;
        Ok(())
    }
}
