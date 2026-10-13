mod dense;
mod gate_act_mul;

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{Backend, CommandBuffer},
    config::mlp::AnyMLPConfig,
    data_type::DataType,
    encodable_block::mlp::dense::{DenseMlp, DenseMlpNewError},
    parameters::ParameterTree,
};

pub trait Mlp<B: Backend>: Send + Sync {
    fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error>;
}

#[derive(Debug, Error)]
pub enum MlpBlockNewError<B: Backend> {
    #[error("Dense MLP error: {0}")]
    Dense(#[from] DenseMlpNewError<B>),
    #[error("Mixture of experts is not supported")]
    UnsupportedMixtureOfExperts,
}

impl<B: Backend> dyn Mlp<B> {
    pub fn new(
        model_dimension: u32,
        hidden_dimension: u32,
        data_type: DataType,
        config: &AnyMLPConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<(Box<dyn Mlp<B>>, Option<B::GlobalBuffer>), MlpBlockNewError<B>> {
        match config {
            AnyMLPConfig::DenseMLPConfig(config) => {
                let (dense, up_input_hadamard_factors) =
                    DenseMlp::new(model_dimension, hidden_dimension, data_type, config, parameter_tree, context)?;

                Ok((Box::new(dense), up_input_hadamard_factors))
            },
            AnyMLPConfig::MixtureOfExpertsConfig(_) => Err(MlpBlockNewError::UnsupportedMixtureOfExperts),
        }
    }
}
