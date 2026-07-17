use std::any::Any;

use thiserror::Error;

use crate::{
    backends::common::{Allocation, Backend, Encoder},
    config::{rope::AnyRoPEConfig, token_mixer::AnyTokenMixerConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        mixer::{
            attention::{Attention, AttentionNewError, rope::PrecalculatedRoPE},
            delta_net::{DeltaNet, DeltaNetNewError},
            mamba2::{Mamba2, Mamba2NewError},
            short_conv::{ShortConv, ShortConvNewError},
        },
    },
    parameters::ParameterTree,
    utils::maybe_mut::MaybeMut,
};

pub mod attention;
pub mod delta_net;
pub mod mamba2;
pub mod short_conv;

pub trait MixerState<B: Backend>: Any + Send {
    fn prepare(
        &mut self,
        context_length: usize,
        suffix_length: usize,
        context: &B::Context,
    ) -> Result<(), B::Error>;

    fn encode_accept(
        &mut self,
        accepted_indices: &[usize],
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error>;
}

impl<'a, B: Backend> MaybeMut<'a, dyn MixerState<B>> {
    pub fn downcast<T: MixerState<B>>(self) -> Option<MaybeMut<'a, T>> {
        match self {
            MaybeMut::Const(value) => (value as &dyn Any).downcast_ref::<T>().map(MaybeMut::Const),
            MaybeMut::Mut(value) => (value as &mut dyn Any).downcast_mut::<T>().map(MaybeMut::Mut),
        }
    }
}

pub trait Mixer<B: Backend>: Send + Sync {
    fn speculation_supported(&self) -> bool;

    fn max_context_length(&self) -> Option<usize>;

    fn create_empty_state(
        &self,
        max_context_length: Option<usize>,
        context: &B::Context,
    ) -> Result<Box<dyn MixerState<B>>, B::Error>;

    fn encode(
        &self,
        hidden: Allocation<B>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error>;
}

#[derive(Debug, Error)]
pub enum MixerNewError<B: Backend> {
    #[error("Attention mixer error: {0}")]
    Attention(#[from] AttentionNewError<B>),
    #[error("Delta net mixer error: {0}")]
    DeltaNet(#[from] DeltaNetNewError<B>),
    #[error("Mamba2 mixer error: {0}")]
    Mamba2(#[from] Mamba2NewError<B>),
    #[error("Short conv mixer error: {0}")]
    ShortConv(#[from] ShortConvNewError<B>),
}

impl<B: Backend> dyn Mixer<B> {
    pub fn new(
        hidden_dim: usize,
        data_type: DataType,
        rope_config: Option<&AnyRoPEConfig>,
        config: &AnyTokenMixerConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<(Box<dyn Mixer<B>>, Option<Allocation<B>>), MixerNewError<B>> {
        match config {
            AnyTokenMixerConfig::AttentionConfig(config) => {
                let (attention, in_projection_input_hadamard_factors) =
                    Attention::new(hidden_dim, data_type, rope_config, config, parameter_tree, context)?;

                Ok((Box::new(attention), in_projection_input_hadamard_factors))
            },
            AnyTokenMixerConfig::DeltaNetConfig(config) => {
                assert!(rope_config.is_none(), "unexpected rope for delta net mixer");

                let (delta_net, in_projection_input_hadamard_factors) =
                    DeltaNet::new(hidden_dim, data_type, config, parameter_tree, context)?;

                Ok((Box::new(delta_net), in_projection_input_hadamard_factors))
            },
            AnyTokenMixerConfig::Mamba2Config(config) => {
                assert!(rope_config.is_none(), "unexpected rope for mamba2 mixer");

                let (mamba2, in_projection_input_hadamard_factors) =
                    Mamba2::new(hidden_dim, data_type, config, parameter_tree, context)?;

                Ok((Box::new(mamba2), in_projection_input_hadamard_factors))
            },
            AnyTokenMixerConfig::ShortConvConfig(config) => {
                assert!(rope_config.is_none(), "unexpected rope for short conv mixer");

                let (short_conv, in_projection_input_hadamard_factors) =
                    ShortConv::new(hidden_dim, data_type, config, parameter_tree, context)?;

                Ok((Box::new(short_conv), in_projection_input_hadamard_factors))
            },
        }
    }
}
