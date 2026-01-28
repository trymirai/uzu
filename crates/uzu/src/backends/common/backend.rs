use std::error::Error;

use super::{Buffer, Context, Device, Kernels};

pub trait Backend {
    type Buffer: Buffer;
    type ResourceOptions: Copy + Send + Sync;
    type Device: Device<Buffer = Self::Buffer, ResourceOptions = Self::ResourceOptions>;
    type Context: Context<Backend = Self>;
    type Buffer;
    type CommandBuffer;
    type Kernels: Kernels<Backend = Self>;
    type Error: Error;
}
