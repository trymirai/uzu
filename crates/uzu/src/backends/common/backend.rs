use std::error::Error;

use super::{Context, Kernels};

pub trait Backend {
    type Context: Context<Backend = Self>;
    type Buffer;
    type CommandBuffer;
    type Kernels: Kernels<Backend = Self>;

    type Error: Error;
}
