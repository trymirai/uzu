use std::{convert::Infallible, sync::Arc};

use super::{
    Error, VkBuffer, VkBufferInitializer, VkCommandBuffer, VkContext, VkDebugGroupSpan, VkSparseBuffer, VulkanKernels,
};
use crate::backends::common::{
    Backend,
    allocator::{
        bump::BumpAllocation,
        pool::{PoolAllocation, PoolAllocator},
    },
};

#[derive(Debug, Clone)]
pub struct Vulkan;

impl Backend for Vulkan {
    type Context = VkContext;
    type CommandBuffer = VkCommandBuffer;
    type DebugGroupSpan<'a> = VkDebugGroupSpan<'a>;
    type GlobalBuffer = Arc<VkBuffer>;
    type GlobalBufferInitializer = VkBufferInitializer;
    type ConstantBuffer = BumpAllocation<Self::GlobalBuffer>;
    type ScratchBuffer = PoolAllocation<Self::GlobalBuffer>;
    type SparseBuffer = VkSparseBuffer;
    type AllocationPool = PoolAllocator<Self::GlobalBuffer>;
    type CaptureSpan = Infallible;
    type Kernels = VulkanKernels;
    type Error = Error;

    const NAME: &'static str = "vulkan";
}
