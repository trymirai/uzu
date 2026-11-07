use crate::backends::{
    common::CommandBuffer,
    vulkan::{
        VkCommandBufferCompleted, VkCommandBufferEncoding, VkCommandBufferExecutable, VkCommandBufferPending, Vulkan,
    },
};

pub struct VkCommandBuffer;

impl CommandBuffer for VkCommandBuffer {
    type Backend = Vulkan;

    type Encoding = VkCommandBufferEncoding;
    type Executable = VkCommandBufferExecutable;
    type Pending = VkCommandBufferPending;
    type Completed = VkCommandBufferCompleted;
}
