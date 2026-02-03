use std::error::Error;

use super::{Context, Kernels, NativeBuffer};

pub trait Backend {
    type Context: Context<Backend = Self>;
    type CommandBuffer;
    type NativeBuffer: NativeBuffer<Backend = Self>;
    type EncoderRef;
    type Kernels: Kernels<Backend = Self>;
    type Error: Error;
}
