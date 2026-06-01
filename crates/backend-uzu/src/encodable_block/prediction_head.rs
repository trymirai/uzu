use thiserror::Error;

use crate::{
    backends::common::{
        Allocation, Backend, Encoder,
        gpu_types::ActivationType,
        kernel::{ActivationKernel, Kernels},
    },
    config::classifier::PredictionHeadConfig,
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearBlockError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar},
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum PredictionHeadError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    Parameter(#[from] ParameterLoaderError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
}

pub struct PredictionHead<B: Backend> {
    hidden_dim: usize,
    activation: ActivationType,
    data_type: DataType,
    dense_projection: Box<dyn Linear<B>>,
    activation_kernel: <B::Kernels as Kernels>::ActivationKernel,
    normalization: Normalization<B>,
    readout: Box<dyn Linear<B>>,
}

impl<B: Backend> PredictionHead<B> {
    pub fn new(
        hidden_dim: usize,
        num_labels: usize,
        data_type: DataType,
        config: &PredictionHeadConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Self, PredictionHeadError<B>> {
        let dense_projection = <dyn Linear<B>>::new(
            hidden_dim,
            [hidden_dim],
            config.use_dense_bias,
            context,
            data_type,
            &parameter_tree.subtree("dense")?,
        )?;

        let activation = config.activation.act_type();
        let activation_kernel = <B::Kernels as Kernels>::ActivationKernel::new(context, data_type, true)
            .map_err(PredictionHeadError::Backend)?;

        let normalization = Normalization::new(
            hidden_dim,
            None,
            false,
            false,
            PostLayerScalar::None,
            data_type,
            &config.normalization_config,
            &parameter_tree.subtree("norm")?,
            context,
        )?;

        let readout = <dyn Linear<B>>::new(
            hidden_dim,
            [num_labels],
            true,
            context,
            data_type,
            &parameter_tree.subtree("readout")?,
        )?;

        Ok(Self {
            hidden_dim,
            activation,
            data_type,
            dense_projection,
            activation_kernel,
            normalization,
            readout,
        })
    }

    pub fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut hidden = self.dense_projection.encode(input, batch_dim, encoder)?;
        self.activation_kernel.encode(
            None::<&Allocation<B>>,
            &mut hidden,
            self.hidden_dim as u32,
            self.activation,
            encoder,
        );
        let normalized = self.normalization.encode(&hidden, 0, batch_dim, None, encoder)?;
        self.readout.encode(normalized, batch_dim, encoder)
    }
}
