//! Mixer component - either attention or state space.

use std::rc::Rc;

use crate::{
    backends::common::Backend,
    encodable_block::{Attention, DeltaNetMixer, MambaMixer, QKNorm, Rope, ShortConvMixer, ValueNorm, linear::Linear},
};

/// Mixer component - either attention, state space, short conv, or delta net.
pub(crate) enum MixerExecutables<B: Backend> {
    Attention {
        qkv_projection: Box<dyn Linear<B>>,
        gate_projection: Option<Box<dyn Linear<B>>>,
        qk_norm: Option<QKNorm<B>>,
        value_norm: Option<ValueNorm<B>>,
        rope: Rc<Rope<B>>,
        use_rope: bool,
        rope_dim: usize,
        num_heads: usize,
        num_groups: usize,
        head_dim: usize,
        attention: Attention<B>,
        out_projection: Box<dyn Linear<B>>,
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
