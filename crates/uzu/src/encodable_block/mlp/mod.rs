mod dense;
mod moe;

pub use dense::DenseMlp;
pub use moe::{MoeBlock, MoeBlockError};
use thiserror::Error;

use super::linear::{Linear, LinearBlockError};
use crate::{
    DataType,
    backends::common::{Backend, Encoder, kernel::mlp_gate_act_mul::MlpGateActMulEncodable},
    config::MLPConfig,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
};

pub trait Mlp<B: Backend> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error>;
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
    ) -> Result<(Box<dyn Mlp<B>>, Option<B::Buffer>), MlpBlockError<B>> {
        if let MLPConfig::Dense(dense_config) = config {
            let data_type: DataType = dense_config.linear_config.activation_precision().into();

            let (up_projection, up_input_hadamard_factors) = <dyn Linear<B>>::new_extracting_input_hadamard(
                &dense_config.linear_config,
                false,
                model_dimension,
                [2 * hidden_dimension],
                context,
                &parameter_tree.subtree("up_projection")?,
                ArrayId::Main,
                ArrayId::MlpFusedUp,
            )?;

            let (down_projection, down_input_hadamard_factors) = <dyn Linear<B>>::new_extracting_input_hadamard(
                &dense_config.linear_config,
                false,
                hidden_dimension,
                [model_dimension],
                context,
                &parameter_tree.subtree("down_projection")?,
                ArrayId::MlpHidden,
                ArrayId::Main,
            )?;

            let gate = MlpGateActMulEncodable::new(
                context,
                data_type,
                dense_config.activation.clone(),
                hidden_dimension,
                down_input_hadamard_factors,
            )
            .map_err(MlpBlockError::BackendError)?;

            return Ok((Box::new(DenseMlp::new(up_projection, gate, down_projection)), up_input_hadamard_factors));
        }

        if let MLPConfig::MixtureOfExperts(mixture_of_experts_config) = config {
            let mixture_of_experts_block =
                MoeBlock::new(context, mixture_of_experts_config, model_dimension, hidden_dimension, parameter_tree)?;
            return Ok((Box::new(mixture_of_experts_block), None));
        }

        unreachable!("Unknown MLP config")
    }
}
