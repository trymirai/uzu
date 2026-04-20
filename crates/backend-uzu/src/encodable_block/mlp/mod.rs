mod dense;
mod moe;

pub use dense::DenseMlp;
pub use moe::{MoeBlock, MoeBlockError};
use thiserror::Error;

use super::linear::{Linear, LinearBlockError};
use crate::{
    DataType,
    backends::common::{Allocation, Backend, Encoder, kernel::mlp_gate_act_mul::MlpGateActMulEncodable},
    config::MLPConfig,
    parameters::{ParameterLoaderError, ParameterTree},
};

pub trait Mlp<B: Backend> {
    fn encode(
        &self,
        input: &mut Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error>;
}

#[derive(Debug, Error)]
pub enum MlpBlockError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Linear block error: {0}")]
    LinearBlockError(#[from] LinearBlockError<B>),
    #[error("MoeBlock error: {0}")]
    MoeBlockError(#[from] MoeBlockError<B>),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
}

impl<B: Backend> dyn Mlp<B> {
    pub fn new(
        config: &MLPConfig,
        model_dimension: usize,
        hidden_dimension: usize,
        context: &B::Context,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<(Box<dyn Mlp<B>>, Option<Allocation<B>>), MlpBlockError<B>> {
        if let MLPConfig::Dense(dense_config) = config {
            let data_type: DataType = dense_config.linear_config.activation_precision().into();

            let (up_projection, up_input_hadamard_factors) = <dyn Linear<B>>::new_extracting_input_hadamard(
                &dense_config.linear_config,
                model_dimension,
                [2 * hidden_dimension],
                context,
                &parameter_tree.subtree("up_projection")?,
            )?;

            let (down_projection, down_input_hadamard_factors) = <dyn Linear<B>>::new_extracting_input_hadamard(
                &dense_config.linear_config,
                hidden_dimension,
                [model_dimension],
                context,
                &parameter_tree.subtree("down_projection")?,
            )?;

            let gate = MlpGateActMulEncodable::new(
                context,
                data_type,
                dense_config.activation.clone(),
                hidden_dimension,
                down_input_hadamard_factors,
            )
            .map_err(MlpBlockError::BackendError)?;

            return Ok((
                Box::new(DenseMlp::new(up_projection, gate, down_projection, hidden_dimension, data_type)),
                up_input_hadamard_factors,
            ));
        }

        if let MLPConfig::MixtureOfExperts(mixture_of_experts_config) = config {
            let mixture_of_experts_block =
                MoeBlock::new(context, mixture_of_experts_config, model_dimension, hidden_dimension, parameter_tree)?;
            return Ok((Box::new(mixture_of_experts_block), None));
        }

        unreachable!("Unknown MLP config")
    }
}
