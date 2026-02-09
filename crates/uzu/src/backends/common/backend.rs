use std::error::Error;

use super::{CommandBuffer, Context, Kernels, NativeBuffer};

pub trait Backend: Clone {
    type Context: Context<Backend = Self>;
    type CommandBuffer: CommandBuffer<Backend = Self>;
    type NativeBuffer: NativeBuffer<Backend = Self>;
    type EncoderRef;
    type Kernels: Kernels<Backend = Self>;
    type Error: Error;
}
