use super::{
    VkCommandBufferCompleted, VkCommandBufferEncoding, VkCommandBufferExecutable, VkCommandBufferPending, Vulkan,
};
use crate::backends::common::CommandBuffer;

pub struct VkCommandBuffer;

impl CommandBuffer for VkCommandBuffer {
    type Backend = Vulkan;

    type Encoding = VkCommandBufferEncoding;
    type Executable = VkCommandBufferExecutable;
    type Pending = VkCommandBufferPending;
    type Completed = VkCommandBufferCompleted;
}
