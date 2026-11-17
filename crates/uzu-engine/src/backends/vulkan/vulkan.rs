use std::{convert::Infallible, sync::Arc};

use crate::backends::{
    common::{
        Backend,
        allocator::{
            bump::BumpAllocation,
            pool::{PoolAllocation, PoolAllocator},
        },
    },
    vulkan::{
        Error, VkBuffer, VkBufferCpuGpu, VkCommandBuffer, VkContext, VkDebugGroupSpan, VkSparseBuffer, VulkanKernels,
    },
};

#[derive(Debug, Clone)]
pub struct Vulkan;

impl Backend for Vulkan {
    type Context = VkContext;
    type CommandBuffer = VkCommandBuffer;
    type DebugGroupSpan<'a> = VkDebugGroupSpan<'a>;
    type GlobalGpuBuffer = Arc<VkBuffer>;
    type GlobalBufferCpuGpu = VkBufferCpuGpu;
    type ConstantBuffer = BumpAllocation<VkBufferCpuGpu>;
    type ScratchBuffer = PoolAllocation<Self::GlobalGpuBuffer>;
    type SparseBuffer = VkSparseBuffer;
    type AllocationPool = PoolAllocator<Self::GlobalGpuBuffer>;
    type CaptureSpan = Infallible;
    type Kernels = VulkanKernels;
    type Error = Error;

    const NAME: &'static str = "vulkan";
}
