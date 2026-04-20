//! MLP block encodable.

use super::{super::linear::Linear, Mlp};
use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{Allocation, Backend, Encoder, kernel::mlp_gate_act_mul::MlpGateActMulEncodable},
};

pub struct DenseMlp<B: Backend> {
    up: Box<dyn Linear<B>>,
    gate: MlpGateActMulEncodable<B>,
    down: Box<dyn Linear<B>>,
    hidden_dim: usize,
    data_type: DataType,
}

impl<B: Backend> DenseMlp<B> {
    pub fn new(
        up: Box<dyn Linear<B>>,
        gate: MlpGateActMulEncodable<B>,
        down: Box<dyn Linear<B>>,
        hidden_dim: usize,
        data_type: DataType,
    ) -> Self {
        Self {
            up,
            gate,
            down,
            hidden_dim,
            data_type,
        }
    }
}

impl<B: Backend> Mlp<B> for DenseMlp<B> {
    fn encode(
        &self,
        context: &B::Context,
        input: &mut Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let fused_up = self.up.encode(context, input, batch_dim, encoder)?;
        debug_assert_eq!(self.gate.hidden_dim(), self.hidden_dim);
        let mut hidden = encoder.allocate_scratch(size_for_shape(&[batch_dim, self.hidden_dim], self.data_type))?;
        self.gate.encode(encoder, &fused_up, &mut hidden, batch_dim as i32)?;
        self.down.encode(context, &mut hidden, batch_dim, encoder)
    }
}
