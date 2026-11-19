use thiserror::Error;

use crate::{
    backends::{
        Backend,
        common::{
            CommandBuffer, CommandBufferEncoding, Kernels, gpu_types::activation_type::ActivationType,
            kernel::ActivationKernel,
        },
    },
    config::vision::UngatedMLPConfig,
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearBlockError},
        mlp::MlpEncodeError,
    },
    safetensors::ParameterTree,
};

#[derive(Debug, Error)]
pub enum UngatedMlpNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
}

pub struct UngatedMlp<B: Backend> {
    activation: <B::Kernels as Kernels>::ActivationKernel,
    up: Box<dyn Linear<B>>,
    down: Box<dyn Linear<B>>,
    hidden_dim: u32,
    activation_type: ActivationType,
    activation_alpha: f32,
}

impl<B: Backend> UngatedMlp<B> {
    pub async fn new(
        data_type: DataType,
        config: &UngatedMLPConfig,
        parameter_tree: &ParameterTree<'_, B>,
        input_dim: u32,
        hidden_dim: u32,
        output_dim: u32,
        context: &B::Context,
    ) -> Result<Self, UngatedMlpNewError<B>> {
        let activation = <B::Kernels as Kernels>::ActivationKernel::new(context, data_type, true)
            .map_err(UngatedMlpNewError::Backend)?;

        let up = <dyn Linear<B>>::new(
            input_dim,
            hidden_dim,
            data_type,
            config.has_up_biases,
            &parameter_tree.subtree("up_projection"),
            context,
        )
        .await?;

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
            activation,
            up,
            down,
            hidden_dim,
            activation_type: config.activation.act_type(),
            activation_alpha: config.activation.alpha(),
        })
    }

    fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, MlpEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("ungated mlp");

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
