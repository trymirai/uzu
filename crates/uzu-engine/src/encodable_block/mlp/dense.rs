//! MLP block encodable.

use crate::{
    backends::common::{Backend, CommandBuffer, CommandBufferEncoding},
    encodable_block::{
        linear::Linear,
        mlp::{Mlp, gate_act_mul::MlpGateActMulEncodable},
    },
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
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut command_buffer = command_buffer.span("mlp");

        let fused_up = self.up.encode(input, batch_dim, &mut command_buffer.span("up projection"))?;
        let act_format = self.down.select_activation_format(batch_dim, command_buffer.context());
        let down_input = self.gate.encode_for_linear(&mut command_buffer, &fused_up, batch_dim, act_format)?;
        self.down.encode_input(down_input, batch_dim, &mut command_buffer.span("down projection"))
    }
}
