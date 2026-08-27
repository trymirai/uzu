use crate::backends::{
    common::{
        Backend,
        allocator::{Allocation, AllocationPool},
    },
    cpu::{
        buffer::{dense::CpuBuffer, sparse::CpuSparseBuffer},
        command_buffer::CpuCommandBuffer,
        context::CpuContext,
        error::CpuError,
        kernel::CpuKernels,
    },
};

#[derive(Debug, Clone)]
pub struct Cpu;

impl Backend for Cpu {
    type Context = CpuContext;
    type CommandBuffer = CpuCommandBuffer;
    type GlobalBuffer = Allocation<CpuBuffer>;
    type ConstantBuffer = Allocation<CpuBuffer>;
    type ScratchBuffer = Allocation<CpuBuffer>;
    type SparseBuffer = CpuSparseBuffer;
    type AllocationPool = AllocationPool<CpuBuffer>;
    type Kernels = CpuKernels;
    type Error = CpuError;

    const NAME: &'static str = "cpu";
}
