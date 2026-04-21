mod dense;
mod moe;

use thiserror::Error;

pub use dense::DenseMlp;
pub use moe::{MoeBlock, MoeBlockError};

use super::linear::{Linear, LinearBlockError};
use crate::{
    DataType,
    backends::common::{Backend, CommandBuffer, kernel::mlp_gate_act_mul::MlpGateActMulEncodable},
    config::MLPConfig,
    encodable_block::EncodingParameters,
    forward_pass::state::ForwardPassState,
    parameters::{ParameterLoaderError, ParameterTree},
};

pub trait Mlp<B: Backend> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
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
    #[error("Invalid dense MLP shapes for `{label}`: {reason}")]
    InvalidDenseMlpShape {
        label: String,
        reason: String,
    },
}

impl<B: Backend> dyn Mlp<B> {
    pub fn new(
        config: &MLPConfig,
        model_dimension: usize,
        hidden_dimension: usize,
        context: &B::Context,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Box<dyn Mlp<B>>, MlpBlockError<B>> {
        match config {
            MLPConfig::Dense(dense_config) => {
                let actual_hidden_dimension =
                    dense_hidden_dimension(parameter_tree, model_dimension, hidden_dimension)?;
                let data_type: DataType = dense_config.linear_config.activation_precision().into();
                let up_projection = <dyn Linear<B>>::new(
                    &dense_config.linear_config,
                    false,
                    model_dimension,
                    [2 * actual_hidden_dimension],
                    context,
                    &parameter_tree.subtree("up_projection")?,
                    crate::forward_pass::state::ArrayId::Main,
                    crate::forward_pass::state::ArrayId::MlpFusedUp,
                )?;
                let gate_activation = MlpGateActMulEncodable::new(
                    context,
                    data_type,
                    dense_config.activation.clone(),
                    actual_hidden_dimension,
                )
                .map_err(MlpBlockError::BackendError)?;
                let down_projection = <dyn Linear<B>>::new(
                    &dense_config.linear_config,
                    false,
                    actual_hidden_dimension,
                    [model_dimension],
                    context,
                    &parameter_tree.subtree("down_projection")?,
                    crate::forward_pass::state::ArrayId::MlpHidden,
                    crate::forward_pass::state::ArrayId::Main,
                )?;
                Ok(Box::new(DenseMlp::new(up_projection, gate_activation, down_projection)))
            },
            MLPConfig::MixtureOfExperts(mixture_of_experts_config) => Ok(Box::new(MoeBlock::new(
                context,
                mixture_of_experts_config,
                model_dimension,
                hidden_dimension,
                parameter_tree,
            )?)),
        }
    }
}

fn dense_hidden_dimension<B: Backend>(
    parameter_tree: &ParameterTree<B::Context>,
    model_dimension: usize,
    configured_hidden_dimension: usize,
) -> Result<usize, MlpBlockError<B>> {
    let label = parameter_tree.path_prefix().unwrap_or("mlp").to_string();
    let up_projection = parameter_tree.subtree("up_projection")?;
    let down_projection = parameter_tree.subtree("down_projection")?;
    let up_shape = up_projection.leaf("weights")?.shape().to_vec();
    let down_shape = down_projection.leaf("weights")?.shape().to_vec();
    let up_bias_shape = match up_projection.leaf("biases") {
        Ok(biases) => Some(biases.shape().to_vec()),
        Err(ParameterLoaderError::KeyNotFound(_)) => None,
        Err(error) => return Err(error.into()),
    };
    let down_bias_shape = match down_projection.leaf("biases") {
        Ok(biases) => Some(biases.shape().to_vec()),
        Err(ParameterLoaderError::KeyNotFound(_)) => None,
        Err(error) => return Err(error.into()),
    };
    let actual_hidden_dimension = dense_hidden_dimension_from_shapes(
        &up_shape,
        up_bias_shape,
        &down_shape,
        down_bias_shape,
        model_dimension,
        configured_hidden_dimension,
    )
    .map_err(|reason| MlpBlockError::InvalidDenseMlpShape {
        label,
        reason,
    })?;
    Ok(actual_hidden_dimension)
}

fn dense_hidden_dimension_from_shapes(
    up_weights_shape: &[usize],
    up_bias_shape: Option<Vec<usize>>,
    down_weights_shape: &[usize],
    down_bias_shape: Option<Vec<usize>>,
    model_dimension: usize,
    configured_hidden_dimension: usize,
) -> Result<usize, String> {
    if up_weights_shape.len() != 2 {
        return Err(format!("up_projection.weights must be 2D, got {up_weights_shape:?}"));
    }
    if up_weights_shape[1] != model_dimension {
        return Err(format!(
            "up_projection.weights second dimension must equal model_dim ({model_dimension}), got {up_weights_shape:?}"
        ));
    }
    if up_weights_shape[0] % 2 != 0 {
        return Err(format!("up_projection.weights first dimension must be even, got {up_weights_shape:?}"));
    }

    let actual_hidden_dimension = up_weights_shape[0] / 2;
    if actual_hidden_dimension == 0 {
        return Err("dense MLP hidden dimension must be non-zero".to_string());
    }
    if actual_hidden_dimension > configured_hidden_dimension {
        return Err(format!(
            "dense MLP hidden dimension ({actual_hidden_dimension}) exceeds configured hidden_dim ({configured_hidden_dimension})"
        ));
    }
    if down_weights_shape != [model_dimension, actual_hidden_dimension] {
        return Err(format!(
            "down_projection.weights must have shape [{model_dimension}, {actual_hidden_dimension}], got {down_weights_shape:?}"
        ));
    }
    if let Some(shape) = up_bias_shape
        && shape != [2 * actual_hidden_dimension]
    {
        return Err(format!("up_projection.biases must have shape [{}], got {shape:?}", 2 * actual_hidden_dimension));
    }
    if let Some(shape) = down_bias_shape
        && shape != [model_dimension]
    {
        return Err(format!("down_projection.biases must have shape [{model_dimension}], got {shape:?}"));
    }
    Ok(actual_hidden_dimension)
}

#[cfg(test)]
mod tests {
    use super::dense_hidden_dimension_from_shapes;

    #[test]
    fn dense_hidden_dimension_accepts_compacted_shapes() {
        assert_eq!(
            dense_hidden_dimension_from_shapes(&[102, 64], Some(vec![102]), &[64, 51], Some(vec![64]), 64, 144)
                .unwrap(),
            51
        );
    }

    #[test]
    fn dense_hidden_dimension_rejects_oversized_compacted_shapes() {
        let error = dense_hidden_dimension_from_shapes(&[290, 64], None, &[64, 145], None, 64, 144).unwrap_err();
        assert!(error.contains("exceeds configured hidden_dim"));
    }

    #[test]
    fn dense_hidden_dimension_rejects_invalid_bias_shape() {
        let error =
            dense_hidden_dimension_from_shapes(&[102, 64], Some(vec![101]), &[64, 51], None, 64, 144).unwrap_err();
        assert!(error.contains("up_projection.biases"));
    }
}
