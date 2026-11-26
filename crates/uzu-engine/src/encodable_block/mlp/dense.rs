//! MLP block encodable.

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{Backend, CommandBuffer, CommandBufferEncoding, gpu_types::activation_type::ActivationType},
    config::{clipping::ClippingBounds, mlp::dense_mlp::DenseMLPConfig},
    data_type::DataType,
    encodable_block::{
        linear::{InputRhtParameters, Linear, LinearBlockError, RhtW4Layer},
        mlp::{Mlp, MlpEncodeError, gate_act_mul::MlpGateActMulEncodable},
    },
    safetensors::ParameterTree,
};

pub struct DenseMlp<B: Backend> {
    up: Box<dyn Linear<B>>,
    gate: MlpGateActMulEncodable<B>,
    down: Box<dyn Linear<B>>,
    /// Plain SiLU, unclipped, in bf16: what a block that restages the projections computes.
    restageable: bool,
}

#[derive(Debug, Error)]
pub enum DenseMlpNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
}

impl<B: Backend> DenseMlp<B> {
    pub async fn new(
        model_dimension: u32,
        hidden_dimension: u32,
        data_type: DataType,
        config: &DenseMLPConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<(Self, Option<InputRhtParameters<B>>), DenseMlpNewError<B>> {
        let (up, up_input_hadamard_factors) = <dyn Linear<B>>::new_with_input_rht(
            model_dimension,
            2 * hidden_dimension,
            data_type,
            config.has_up_biases,
            &parameter_tree.subtree("up_projection"),
            context,
        )
        .await?;

        let (down, down_input_preparation) = <dyn Linear<B>>::new_for_fused_input(
            hidden_dimension,
            model_dimension,
            data_type,
            config.has_down_biases,
            &parameter_tree.subtree("down_projection"),
            context,
        )
        .await?;

        let gate = MlpGateActMulEncodable::new(
            hidden_dimension,
            data_type,
            config.activation.clone(),
            config.gate_clipping,
            config.up_clipping,
            down_input_preparation,
            context,
        )
        .map_err(DenseMlpNewError::Backend)?;

        let restageable = data_type == DataType::BF16
            && config.activation.act_type() == ActivationType::SILU
            && config.activation.custom_alpha().is_none()
            && config.gate_clipping == ClippingBounds::default()
            && config.up_clipping == ClippingBounds::default();

        Ok((
            Self {
                up,
                gate,
                down,
                restageable,
            },
            up_input_hadamard_factors,
        ))
    }

    /// The up and down projections, if both are RHT W4 layers of a plain bf16 SiLU MLP.
    pub fn rht_w4(&self) -> Option<(RhtW4Layer<'_, B>, RhtW4Layer<'_, B>)> {
        if !self.restageable {
            return None;
        }
        Some((self.up.rht_w4(None)?, self.down.rht_w4(self.gate.input_rht())?))
    }
}

impl<B: Backend> Mlp<B> for DenseMlp<B> {
    fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, MlpEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("mlp (dense)");

        let fused_up = self.up.encode(input, batch_dim, command_buffer)?;
        let act_format = self.down.select_activation_format(batch_dim, None, command_buffer.context());
        let down_input = self
            .gate
            .encode_for_linear(&fused_up, batch_dim, act_format, command_buffer)
            .map_err(MlpEncodeError::Backend)?;
        let output = self.down.encode_input(down_input, batch_dim, command_buffer)?;

        Ok(output)
    }
}
