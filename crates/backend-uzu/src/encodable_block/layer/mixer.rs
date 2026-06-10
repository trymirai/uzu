//! Mixer component - either attention or state space.

use crate::{
    backends::common::Backend,
    encodable_block::{Attention, DeltaNetMixer, MambaMixer, ShortConvMixer},
};

/// Mixer component - either attention, state space, short conv, or delta net.
pub enum MixerExecutables<B: Backend> {
    Attention {
        attention: Attention<B>,
    },
    StateSpace {
        mixer: MambaMixer<B>,
    },
    ShortConv {
        mixer: ShortConvMixer<B>,
    },
    DeltaNet {
        mixer: DeltaNetMixer<B>,
    },
}

impl<B: Backend> MixerExecutables<B> {
    pub fn precompile(
        &self,
        context: &B::Context,
        batch_sizes: &[u32],
    ) -> Result<(), B::Error> {
        match self {
            MixerExecutables::Attention {
                attention,
            } => attention.precompile(context, batch_sizes),
            MixerExecutables::StateSpace {
                ..
            }
            | MixerExecutables::ShortConv {
                ..
            }
            | MixerExecutables::DeltaNet {
                ..
            } => Ok(()),
        }
    }
}
