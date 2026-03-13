//! Mixer component - either attention or state space.

use std::rc::Rc;

use crate::{
    backends::common::Backend,
    encodable_block::{Attention, MambaMixer, QKNorm, Rope, ShortConvMixer, linear::Linear},
};

/// Mixer component - either attention, state space, or short conv.
pub(crate) enum MixerExecutables<B: Backend> {
    Attention {
        qkv_projection: Box<dyn Linear<B>>,
        qk_norm: Option<QKNorm<B>>,
        rope: Rc<Rope<B>>,
        attention: Attention<B>,
        out_projection: Box<dyn Linear<B>>,
    },
    StateSpace {
        mixer: MambaMixer<B>,
    },
    ShortConv {
        mixer: ShortConvMixer<B>,
    },
}
