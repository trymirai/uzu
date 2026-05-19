//! Mixer component - either attention or state space.

use crate::{
    backends::common::Backend,
    encodable_block::{Attention, DeltaNetMixer, MambaMixer, ShortConvMixer},
};

/// Mixer component - either attention, state space, short conv, or delta net.
pub(crate) enum MixerExecutables<B: Backend> {
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
