use std::{error::Error, ops::DerefMut};

use super::{
    Buffer, CommandBuffer, Context, CpuBuffer, GlobalBufferCpuGpu, GlobalGpuBuffer, GpuBuffer, Kernels, SparseBuffer,
};

pub trait Backend: Clone + Send + Sync + 'static {
    type Context: Context<Backend = Self>;
    type CommandBuffer: CommandBuffer<Backend = Self>;
    type DebugGroupSpan<'a>: DerefMut<Target = <Self::CommandBuffer as CommandBuffer>::Encoding> + 'a;
    type GlobalGpuBuffer: GlobalGpuBuffer<Backend = Self>;
    type GlobalBufferCpuGpu: GlobalBufferCpuGpu<Backend = Self>;
    type ConstantBuffer: Buffer<Backend = Self> + GpuBuffer + CpuBuffer;
    type ScratchBuffer: GpuBuffer<Backend = Self>;
    type SparseBuffer: SparseBuffer<Backend = Self>;
    type AllocationPool: Send + Sync;
    type CaptureSpan: Send + Sync;
    type Kernels: Kernels<Backend = Self>;
    type Error: Error;

    const NAME: &'static str;
}
