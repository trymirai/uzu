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
    type Mixer: Mixer<B, State = Self>;

    fn prepare(
        &mut self,
        context_length: u32,
        suffix_length: u32,
        context: &B::Context,
    ) -> Result<(), B::Error>;

    fn encode_accept(
        &mut self,
        accepted_indices: &[u32],
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error>;
}

pub trait DynMixerState<B: Backend>: Any + Send {
    fn prepare(
        &mut self,
        context_length: u32,
        suffix_length: u32,
        context: &B::Context,
    ) -> Result<(), B::Error>;

    fn encode_accept(
        &mut self,
        accepted_indices: &[u32],
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error>;
}

impl<B: Backend, T: MixerState<B>> DynMixerState<B> for T {
    fn prepare(
        &mut self,
        context_length: u32,
        suffix_length: u32,
        context: &B::Context,
    ) -> Result<(), B::Error> {
        MixerState::prepare(self, context_length, suffix_length, context)
    }

    fn encode_accept(
        &mut self,
        accepted_indices: &[u32],
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        MixerState::encode_accept(self, accepted_indices, encoder)
    }
}

impl<'a, B: Backend> MaybeMut<'a, dyn DynMixerState<B>> {
    pub fn downcast<T: MixerState<B>>(self) -> Option<MaybeMut<'a, T>> {
        match self {
            MaybeMut::Const(value) => (value as &dyn Any).downcast_ref::<T>().map(MaybeMut::Const),
            MaybeMut::Mut(value) => (value as &mut dyn Any).downcast_mut::<T>().map(MaybeMut::Mut),
        }
    }
}

pub trait Mixer<B: Backend>: Any + Send + Sync {
    type State: MixerState<B, Mixer = Self>;

    fn speculation_supported(&self) -> bool;

    fn max_context_length(&self) -> Option<u32>;

    fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<Self::State, B::Error>;

    fn encode(
        &self,
        hidden: Allocation<B>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<Self::State>>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error>;
}

pub trait DynMixer<B: Backend>: Any + Send + Sync {
    fn speculation_supported(&self) -> bool;

    fn max_context_length(&self) -> Option<u32>;

    fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<Box<dyn DynMixerState<B>>, B::Error>;

    fn encode(
        &self,
        hidden: Allocation<B>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<dyn DynMixerState<B>>>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error>;
}

impl<B: Backend, T: Mixer<B>> DynMixer<B> for T {
    fn speculation_supported(&self) -> bool {
        Mixer::speculation_supported(self)
    }

    fn max_context_length(&self) -> Option<u32> {
        Mixer::max_context_length(self)
    }

    fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<Box<dyn DynMixerState<B>>, B::Error> {
        Ok(Box::new(Mixer::create_empty_state(self, max_context_length, context)?))
    }

    fn encode(
        &self,
        hidden: Allocation<B>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<dyn DynMixerState<B>>>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let state = state.map(|state| {
            state
                .downcast::<T::State>()
                .unwrap_or_else(|| panic!("incorrect mixer state type: expected {}", std::any::type_name::<T::State>()))
        });
        Mixer::encode(self, hidden, precalculated_rope, batch_dim, state, encoder)
    }
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

impl<B: Backend> dyn DynMixer<B> {
    pub fn new(
        hidden_dim: u32,
        data_type: DataType,
        rope_config: Option<&AnyRoPEConfig>,
        config: &AnyTokenMixerConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<(Box<dyn DynMixer<B>>, Option<Allocation<B>>), MixerNewError<B>> {
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
