//! Shared MoE expert weights.

use std::rc::Rc;

#[derive(Clone)]
pub struct SharedMoeWeights {
    pub w13_buf: Rc<metal::Buffer>,
    pub w2_buf: Rc<metal::Buffer>,
    pub up_biases_buf: Rc<metal::Buffer>,
    pub down_biases_buf: Rc<metal::Buffer>,
}
