//! MLP block encodable.

use super::{super::linear::Linear, Mlp};
use crate::backends::common::{Allocation, Backend, Encoder, kernel::mlp_gate_act_mul::MlpGateActMulEncodable};

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
        input: Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let fused_up = self.up.encode(input, batch_dim, encoder)?;
        let hidden = self.gate.encode(encoder, &fused_up, batch_dim)?;
        self.down.encode(hidden, batch_dim, encoder)
    }
}
