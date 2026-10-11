use std::sync::Arc;

use super::{Error, VkBuffer, VkCommandBuffer, VkContext, VkSparseBuffer, VulkanKernels};
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
    type GlobalBuffer = Arc<VkBuffer>;
    type ConstantBuffer = BumpAllocation<Self::GlobalBuffer>;
    type ScratchBuffer = PoolAllocation<Self::GlobalBuffer>;
    type SparseBuffer = VkSparseBuffer;
    type AllocationPool = PoolAllocator<Self::GlobalBuffer>;
    type Kernels = VulkanKernels;
    type Error = Error;

    const NAME: &'static str = "vulkan";
}
