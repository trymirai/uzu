mod dense;
mod gate_act_mul;

pub use dense::DenseMlp;
use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{Backend, CommandBuffer, Context},
    config::mlp::AnyMLPConfig,
    data_type::DataType,
    encodable_block::{
        linear::{InputBasis, InputRhtParameters, LinearEncodeError},
        mlp::dense::DenseMlpNewError,
    },
    safetensors::ParameterTree,
};

#[derive(Debug, Error)]
pub enum MlpEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearEncodeError<B>),
    #[error("the selected MLP input basis is no longer available")]
    InputBasisUnavailable,
}

/// An MLP whose encoding was started with [`Mlp::encode_start`].
pub enum MlpStarted<'a, B: Backend> {
    /// All of it is encoded.
    Done(B::ScratchBuffer),
    /// Encoded up to a hand-off to other hardware: what is encoded before the closure runs alongside
    /// the hand-off, and the closure encodes the rest and yields the output.
    Handoff(
        Box<
            dyn FnOnce(
                    &mut <B::CommandBuffer as CommandBuffer>::Encoding,
                ) -> Result<B::ScratchBuffer, MlpEncodeError<B>>
                + 'a,
        >,
    ),
}

impl<B: Backend> MlpStarted<'_, B> {
    pub fn finish(
        self,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, MlpEncodeError<B>> {
        match self {
            Self::Done(output) => Ok(output),
            Self::Handoff(finish) => finish(command_buffer),
        }
    }
}

pub trait Mlp<B: Backend>: Send + Sync {
    fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, MlpEncodeError<B>>;

    /// Encodes the MLP up to a hand-off to other hardware (see [`MlpStarted`]). Default: all of it,
    /// as [`Self::encode`].
    fn encode_start<'a>(
        &'a self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<MlpStarted<'a, B>, MlpEncodeError<B>> {
        Ok(MlpStarted::Done(self.encode(input, batch_dim, command_buffer)?))
    }

    fn input_basis(
        &self,
        _batch_dim: u32,
    ) -> InputBasis {
        InputBasis::GpuExpected
    }

    fn supports_unrotated_input(&self) -> bool {
        false
    }

    fn encode_start_for_basis<'a>(
        &'a self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        basis: InputBasis,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<MlpStarted<'a, B>, MlpEncodeError<B>> {
        if basis == InputBasis::AneUnrotated {
            return Err(MlpEncodeError::InputBasisUnavailable);
        }
        self.encode_start(input, batch_dim, command_buffer)
    }
}

#[derive(Debug, Error)]
pub enum MlpBlockNewError<B: Backend> {
    #[error("Dense MLP error: {0}")]
    Dense(#[from] DenseMlpNewError<B>),
    #[error("Mixture of experts is not supported")]
    UnsupportedMixtureOfExperts,
}

impl<B: Backend> dyn Mlp<B> {
    pub async fn new(
        model_dimension: u32,
        hidden_dimension: u32,
        data_type: DataType,
        config: &AnyMLPConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<(Box<dyn Mlp<B>>, Option<InputRhtParameters<B>>), MlpBlockNewError<B>> {
        match config {
            AnyMLPConfig::DenseMLPConfig(config) => {
                let (dense, up_input_hadamard_factors) =
                    DenseMlp::new(model_dimension, hidden_dimension, data_type, config, parameter_tree, context)
                        .await?;

                Ok((context.wrap_mlp(dense), up_input_hadamard_factors))
            },
            AnyMLPConfig::MixtureOfExpertsConfig(_) => Err(MlpBlockNewError::UnsupportedMixtureOfExperts),
        }
    }
}
