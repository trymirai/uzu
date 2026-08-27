//! MLP block encodable.

use crate::{
    backends::common::{Allocation, Backend, CommandBuffer, CommandBufferEncoding},
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
        input: Allocation<B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Allocation<B>, B::Error> {
        command_buffer.push_debug_group("mlp (dense)");

        let fused_up = self.up.encode(input, batch_dim, command_buffer)?;
        let act_format = self.down.select_activation_format(batch_dim, command_buffer.context());
        let down_input = self.gate.encode_for_linear(command_buffer, &fused_up, batch_dim, act_format)?;
        let output = self.down.encode_input(down_input, batch_dim, command_buffer)?;

        command_buffer.pop_debug_group();

        Ok(output)
    }
}
