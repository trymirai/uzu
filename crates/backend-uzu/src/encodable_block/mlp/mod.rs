mod dense;
mod gate_act_mul;
mod moe;

pub use dense::DenseMlp;
use gate_act_mul::MlpGateActMulEncodable;
pub(crate) use moe::MoeRouterScaling;
pub use moe::{MoeBlock, MoeBlockError};
use thiserror::Error;

use super::linear::{Linear, LinearBlockError};
use crate::{
    backends::common::{Allocation, Backend, Encoder},
    config::mlp::AnyMLPConfig,
    data_type::DataType,
    parameters::{ParameterLoaderError, ParameterTree},
};

pub trait Mlp<B: Backend> {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error>;

    fn encode_with_router_input(
        &self,
        _router_input: &Allocation<B>,
        expert_input: Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        self.encode(expert_input, batch_dim, encoder)
    }
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
        config: &AnyMLPConfig,
        model_dimension: usize,
        hidden_dimension: usize,
        context: &B::Context,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<(Box<dyn Mlp<B>>, Option<Allocation<B>>), MlpBlockError<B>> {
        match config {
            AnyMLPConfig::DenseMLPConfig(dense_config) => {
                let (up_projection, up_input_hadamard_factors) = <dyn Linear<B>>::new_extracting_input_hadamard(
                    model_dimension,
                    [2 * hidden_dimension],
                    dense_config.has_up_biases,
                    context,
                    data_type,
                    &parameter_tree.subtree("up_projection")?,
                )?;

                let (down_projection, down_input_hadamard_factors) = <dyn Linear<B>>::new_extracting_input_hadamard(
                    hidden_dimension,
                    [model_dimension],
                    dense_config.has_down_biases,
                    context,
                    data_type,
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

                Ok((Box::new(DenseMlp::new(up_projection, gate, down_projection)), up_input_hadamard_factors))
            },
            AnyMLPConfig::MixtureOfExpertsConfig(mixture_of_experts_config) => Ok((
                Box::new(MoeBlock::new(
                    context,
                    mixture_of_experts_config,
                    model_dimension,
                    data_type,
                    parameter_tree,
                    MoeRouterScaling::default(),
                )?),
                None,
            )),
        }
    }
}
