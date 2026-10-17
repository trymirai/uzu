use std::time::Duration;

use super::VkCommandBuffer;
use crate::backends::common::CommandBufferCompleted;

/// A finished submission; its command resources were already recycled.
pub struct VkCommandBufferCompleted {
    gpu_execution_time: Duration,
}

impl VkCommandBufferCompleted {
    pub fn new(gpu_execution_time: Duration) -> Self {
        Self {
            gpu_execution_time,
        }
    }
}

impl CommandBufferCompleted for VkCommandBufferCompleted {
    type CommandBuffer = VkCommandBuffer;

    fn gpu_execution_time(&self) -> Duration {
        self.gpu_execution_time
    }
}
