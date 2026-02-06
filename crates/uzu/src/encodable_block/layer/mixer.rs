//! Mixer component - either attention or state space.

use std::rc::Rc;

use super::super::EncodableBlock;
use crate::backends::metal::Metal;

/// Mixer component - either attention, state space, or short conv.
pub(crate) enum MixerExecutables {
    Attention {
        qkv_projection: Box<dyn EncodableBlock<Metal>>,
        qk_norm: Option<Box<dyn EncodableBlock<Metal>>>,
        rope: Rc<Box<dyn EncodableBlock<Metal>>>,
        attention: Box<dyn EncodableBlock<Metal>>,
        out_projection: Box<dyn EncodableBlock<Metal>>,
    },
    StateSpace {
        mixer: Box<dyn EncodableBlock<Metal>>,
    },
    ShortConv {
        mixer: Box<dyn EncodableBlock<Metal>>,
    },
}
