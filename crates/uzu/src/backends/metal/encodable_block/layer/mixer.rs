//! Mixer component - either attention or state space.

use std::rc::Rc;

use crate::{backends::common::Backend, encodable_block::EncodableBlock};

/// Mixer component - either attention, state space, or short conv.
pub(crate) enum MixerExecutables<B: Backend> {
    Attention {
        qkv_projection: Box<dyn EncodableBlock<B>>,
        qk_norm: Option<Box<dyn EncodableBlock<B>>>,
        rope: Rc<Box<dyn EncodableBlock<B>>>,
        attention: Box<dyn EncodableBlock<B>>,
        out_projection: Box<dyn EncodableBlock<B>>,
    },
    StateSpace {
        mixer: Box<dyn EncodableBlock<B>>,
    },
    ShortConv {
        mixer: Box<dyn EncodableBlock<B>>,
    },
}
