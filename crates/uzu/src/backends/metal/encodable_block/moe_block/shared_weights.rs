//! Shared MoE expert weights.

use std::rc::Rc;

use crate::backends::metal::Buffer;

#[derive(Clone)]
pub struct SharedMoeWeights {
    pub w13_buf: Rc<Buffer>,
    pub w2_buf: Rc<Buffer>,
    pub up_biases_buf: Rc<Buffer>,
    pub down_biases_buf: Rc<Buffer>,
}
