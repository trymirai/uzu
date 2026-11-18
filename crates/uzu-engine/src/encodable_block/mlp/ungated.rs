use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, CommandBuffer, CommandBufferEncoding,
        gpu_types::activation_type::ActivationType,
        kernel::{ActivationKernel, Kernels},
    },
    config::mlp::ungated::UngatedMLPConfig,
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearBlockError},
        mlp::{Mlp, MlpEncodeError},
    },
    safetensors::ParameterTree,
};

pub struct UngatedMlp<B: Backend> {
    hidden_dim: u32,
    activation_type: ActivationType,
    activation_alpha: f32,
    up: Box<dyn Linear<B>>,
    activation: <B::Kernels as Kernels>::ActivationKernel,
    down: Box<dyn Linear<B>>,
}

#[derive(Debug, Error)]
pub enum UngatedMlpNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
}

impl<B: Backend> UngatedMlp<B> {
    pub async fn new(
        input_dim: u32,
        hidden_dim: u32,
        output_dim: u32,
        data_type: DataType,
        config: &UngatedMLPConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, UngatedMlpNewError<B>> {
        let up = <dyn Linear<B>>::new(
            input_dim,
            hidden_dim,
            data_type,
            config.has_up_biases,
            &parameter_tree.subtree("up_projection"),
            context,
        )
        .await?;

        let activation = <B::Kernels as Kernels>::ActivationKernel::new(context, data_type, true)
            .map_err(UngatedMlpNewError::Backend)?;

        let down = <dyn Linear<B>>::new(
            hidden_dim,
            output_dim,
            data_type,
            config.has_down_biases,
            &parameter_tree.subtree("down_projection"),
            context,
        )
        .await?;

        Ok(Self {
            hidden_dim,
            activation_type: config.activation.act_type(),
            activation_alpha: config.activation.alpha(),
            up,
            activation,
            down,
        })
    }
}

impl<B: Backend> Mlp<B> for UngatedMlp<B> {
    fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, MlpEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("mlp (ungated)");

        let mut hidden = self.up.encode(input, batch_dim, command_buffer)?;
        self.activation.encode(
            None::<&B::ScratchBuffer>,
            &mut hidden,
            batch_dim * self.hidden_dim,
            self.activation_type,
            self.activation_alpha,
            command_buffer,
        );
        let output = self.down.encode(hidden, batch_dim, command_buffer)?;

        Ok(output)
    }
}
