use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, CommandBuffer, CommandBufferEncoding,
        gpu_types::activation_type::ActivationType,
        kernel::{ActivationKernel, Kernels},
    },
    config::classifier::PredictionHeadConfig,
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearBlockError, LinearEncodeError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
    },
    safetensors::ParameterTree,
};

#[derive(Debug, Error)]
pub enum PredictionHeadError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Linear encoding error: {0}")]
    LinearEncode(#[from] LinearEncodeError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
}

pub struct PredictionHead<B: Backend> {
    hidden_dim: u32,
    activation: ActivationType,
    dense_projection: Box<dyn Linear<B>>,
    activation_kernel: <B::Kernels as Kernels>::ActivationKernel,
    normalization: Normalization<B>,
    readout: Box<dyn Linear<B>>,
}

impl<B: Backend> PredictionHead<B> {
    pub async fn new(
        hidden_dim: u32,
        num_labels: u32,
        data_type: DataType,
        config: &PredictionHeadConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, PredictionHeadError<B>> {
        let dense_projection = <dyn Linear<B>>::new(
            hidden_dim,
            hidden_dim,
            data_type,
            config.use_dense_bias,
            &parameter_tree.subtree("dense"),
            context,
        )
        .await?;

        let activation = config.activation.act_type();
        let activation_kernel = <B::Kernels as Kernels>::ActivationKernel::new(context, data_type, true)
            .map_err(PredictionHeadError::Backend)?;

        let normalization = Normalization::new(
            hidden_dim,
            data_type,
            None,
            ShortcutMode::None,
            PostLayerScalar::None,
            &config.normalization_config,
            &parameter_tree.subtree("norm"),
            context,
        )
        .await?;

        let readout =
            <dyn Linear<B>>::new(hidden_dim, num_labels, data_type, true, &parameter_tree.subtree("readout"), context)
                .await?;

        Ok(Self {
            hidden_dim,
            activation,
            dense_projection,
            activation_kernel,
            normalization,
            readout,
        })
    }

    pub fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, PredictionHeadError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("prediction head");

        let mut hidden = self.dense_projection.encode(input, batch_dim, command_buffer)?;
        self.activation_kernel.encode(
            None::<&B::ScratchBuffer>,
            &mut hidden,
            self.hidden_dim,
            self.activation,
            command_buffer,
        );
        let normalized = self
            .normalization
            .encode(&hidden, 0, batch_dim, None::<&mut B::ScratchBuffer>, command_buffer)
            .map_err(PredictionHeadError::Backend)?;
        let logits = self.readout.encode(normalized, batch_dim, command_buffer)?;

        Ok(logits)
    }
}
