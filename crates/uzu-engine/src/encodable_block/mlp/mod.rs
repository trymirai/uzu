mod dense;
mod gate_act_mul;
mod moe;

pub use dense::DenseMlp;
use gate_act_mul::MlpGateActMulEncodable;
pub use moe::{MoeBlock, MoeBlockError};
use thiserror::Error;

use crate::{
    backends::common::{Allocation, Backend, Encoder},
    config::mlp::AnyMLPConfig,
    data_type::DataType,
    encodable_block::linear::{Linear, LinearBlockError},
    parameters::ParameterTree,
};

pub trait Mlp<B: Backend>: Send + Sync {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: u32,
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
}

impl<B: Backend> dyn Mlp<B> {
    pub fn new(
        config: &AnyMLPConfig,
        model_dimension: u32,
        hidden_dimension: u32,
        context: &B::Context,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<(Box<dyn Mlp<B>>, Option<Allocation<B>>), MlpBlockError<B>> {
        match config {
            AnyMLPConfig::DenseMLPConfig(dense_config) => {
                let (up_projection, up_input_hadamard_factors) = <dyn Linear<B>>::new_with_input_rht(
                    model_dimension,
                    [2 * hidden_dimension],
                    dense_config.has_up_biases,
                    context,
                    data_type,
                    &parameter_tree.subtree("up_projection"),
                )?;

                let (down_projection, down_input_preparation) = <dyn Linear<B>>::new_for_fused_input(
                    hidden_dimension,
                    [model_dimension],
                    dense_config.has_down_biases,
                    context,
                    data_type,
                    &parameter_tree.subtree("down_projection"),
                )?;

                let gate = MlpGateActMulEncodable::new(
                    context,
                    data_type,
                    dense_config.activation.clone(),
                    dense_config.gate_clipping,
                    dense_config.up_clipping,
                    hidden_dimension,
                    down_input_preparation,
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
                )?),
                None,
            )),
        }
    }
}
