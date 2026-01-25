//! Shared MoE expert weights.

use std::rc::Rc;

use crate::backends::metal::{MTLBuffer, ProtocolObject, Retained};

#[derive(Clone)]
pub struct SharedMoeWeights {
    pub w13_buf: Rc<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub w2_buf: Rc<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub up_biases_buf: Rc<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub down_biases_buf: Rc<Retained<ProtocolObject<dyn MTLBuffer>>>,
}
