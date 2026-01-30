use std::error::Error;

use super::{Context, Kernels, NativeBuffer};

pub trait Backend {
    type NativeBuffer: NativeBuffer;
    type Context: Context<Backend = Self>;
    type CommandBuffer;
    type BufferRef;
    type EncoderRef;
    type Kernels: Kernels<Backend = Self>;
    type Error: Error;
}
