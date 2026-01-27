use std::error::Error;

use super::{Context, Kernels};

pub trait Backend {
    type Context: Context;
    type CommandBuffer;
    type Kernels: Kernels;

    type Error: Error;
}
