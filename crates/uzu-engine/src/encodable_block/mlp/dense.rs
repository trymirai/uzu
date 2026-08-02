//! MLP block encodable.

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{Backend, CommandBuffer, CommandBufferEncoding},
    config::mlp::dense_mlp::DenseMLPConfig,
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearBlockError},
        mlp::{Mlp, MlpEncodeError, gate_act_mul::MlpGateActMulEncodable},
    },
    safetensors::ParameterTree,
};

pub struct DenseMlp<B: Backend> {
    up: Box<dyn Linear<B>>,
    gate: MlpGateActMulEncodable<B>,
    down: Box<dyn Linear<B>>,
}

#[derive(Debug, Error)]
pub enum DenseMlpNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
}

impl<B: Backend> DenseMlp<B> {
    pub async fn new(
        model_dimension: u32,
        hidden_dimension: u32,
        data_type: DataType,
        config: &DenseMLPConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<(Self, Option<B::GlobalGpuBuffer>), DenseMlpNewError<B>> {
        let (up, up_input_hadamard_factors) = <dyn Linear<B>>::new_with_input_rht(
            model_dimension,
            2 * hidden_dimension,
            data_type,
            config.has_up_biases,
            &parameter_tree.subtree("up_projection"),
            context,
        )
        .await?;

        let (down, down_input_preparation) = <dyn Linear<B>>::new_for_fused_input(
            hidden_dimension,
            model_dimension,
            data_type,
            config.has_down_biases,
            &parameter_tree.subtree("down_projection"),
            context,
        )
        .await?;

        let gate = MlpGateActMulEncodable::new(
            hidden_dimension,
            data_type,
            config.activation.clone(),
            config.gate_clipping,
            config.up_clipping,
            down_input_preparation,
            context,
        )
        .map_err(DenseMlpNewError::Backend)?;

        Ok((
            Self {
                up,
                gate,
                down,
            },
            up_input_hadamard_factors,
        ))
    }
}

impl<B: Backend> Mlp<B> for DenseMlp<B> {
    fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, MlpEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("mlp (dense)");

        let fused_up = self.up.encode(input, batch_dim, command_buffer)?;
        let act_format = self.down.select_activation_format(batch_dim, command_buffer.context());
        let down_input = self
            .gate
            .encode_for_linear(&fused_up, batch_dim, act_format, command_buffer)
            .map_err(MlpEncodeError::Backend)?;
        let output = self.down.encode_input(down_input, batch_dim, command_buffer)?;

        Ok(output)
    }
}
