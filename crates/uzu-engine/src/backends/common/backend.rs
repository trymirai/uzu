use std::{error::Error, ops::DerefMut};

use super::{
    CommandBuffer, ConstantBuffer, Context, GlobalBuffer, GlobalBufferInitializer, Kernels, ScratchBuffer, SparseBuffer,
};

pub trait Backend: Clone + Send + Sync + 'static {
    type Context: Context<Backend = Self>;
    type CommandBuffer: CommandBuffer<Backend = Self>;
    type DebugGroupSpan<'a>: DerefMut<Target = <Self::CommandBuffer as CommandBuffer>::Encoding> + 'a;
    type GlobalBuffer: GlobalBuffer<Backend = Self>;
    type GlobalBufferInitializer: GlobalBufferInitializer<Backend = Self>; // TODO: remove this horrible thing
    type ConstantBuffer: ConstantBuffer<Backend = Self>;
    type ScratchBuffer: ScratchBuffer<Backend = Self>;
    type SparseBuffer: SparseBuffer<Backend = Self>;
    type AllocationPool: Send + Sync;
    type CaptureSpan: Send + Sync;
    type Kernels: Kernels<Backend = Self>;
    type Error: Error;

    const NAME: &'static str;
}
