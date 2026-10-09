mod dense;
mod gate_act_mul;

pub use dense::DenseMlp;
use derive_more::Debug;
use gate_act_mul::MlpGateActMulEncodable;
use thiserror::Error;

use crate::{
    backends::common::{Backend, CommandBuffer},
    config::mlp::AnyMLPConfig,
    data_type::DataType,
    encodable_block::linear::{Linear, LinearBlockError},
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
pub enum MlpBlockError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Linear block error: {0}")]
    LinearBlockError(#[from] LinearBlockError<B>),
    #[error("Mixture of experts is not supported")]
    UnsupportedMixtureOfExperts,
}

impl<B: Backend> dyn Mlp<B> {
    pub fn new(
        name: String,
        config: &AnyMLPConfig,
        model_dimension: u32,
        hidden_dimension: u32,
        context: &B::Context,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<(Box<dyn Mlp<B>>, Option<B::GlobalBuffer>), MlpBlockError<B>> {
        match config {
            AnyMLPConfig::DenseMLPConfig(dense_config) => {
                let (up_projection, up_input_hadamard_factors) = <dyn Linear<B>>::new_with_input_rht(
                    format!("{name}/up projection"),
                    model_dimension,
                    [2 * hidden_dimension],
                    dense_config.has_up_biases,
                    context,
                    data_type,
                    &parameter_tree.subtree("up_projection"),
                )?;

                let (down_projection, down_input_preparation) = <dyn Linear<B>>::new_for_fused_input(
                    format!("{name}/down projection"),
                    hidden_dimension,
                    [model_dimension],
                    dense_config.has_down_biases,
                    context,
                    data_type,
                    &parameter_tree.subtree("down_projection"),
                )?;

                let gate = MlpGateActMulEncodable::new(
                    format!("{name}/gate act mul"),
                    context,
                    data_type,
                    dense_config.activation.clone(),
                    dense_config.gate_clipping,
                    dense_config.up_clipping,
                    hidden_dimension,
                    down_input_preparation,
                )
                .map_err(MlpBlockError::BackendError)?;

                Ok((Box::new(DenseMlp::new(name, up_projection, gate, down_projection)), up_input_hadamard_factors))
            },
            AnyMLPConfig::MixtureOfExpertsConfig(_) => Err(MlpBlockError::UnsupportedMixtureOfExperts),
        }
    }
}
