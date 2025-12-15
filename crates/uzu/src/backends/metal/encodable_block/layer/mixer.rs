//! Mixer component - either attention or state space.

use std::rc::Rc;

use super::super::EncodableBlock;

/// Mixer component - either attention, state space, or short conv.
pub(crate) enum MixerExecutables {
    Attention {
        qkv_projection: Box<dyn EncodableBlock>,
        qk_norm: Option<Box<dyn EncodableBlock>>,
        rope: Rc<Box<dyn EncodableBlock>>,
        attention: Box<dyn EncodableBlock>,
        out_projection: Box<dyn EncodableBlock>,
    },
    StateSpace {
        mixer: Box<dyn EncodableBlock>,
    },
    ShortConv {
        mixer: Box<dyn EncodableBlock>,
    },
}
