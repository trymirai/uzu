use std::time::Duration;

use crate::backends::{
    common::{CommandBufferCompleted, TimestampSpan},
    vulkan::VkCommandBuffer,
};

/// A finished submission; its command resources were already recycled.
pub struct VkCommandBufferCompleted {
    gpu_execution_time: Duration,
    timestamps: Box<[TimestampSpan]>,
}

impl VkCommandBufferCompleted {
    pub fn new(
        gpu_execution_time: Duration,
        timestamps: Box<[TimestampSpan]>,
    ) -> Self {
        Self {
            gpu_execution_time,
            timestamps,
        }
    }
}

impl CommandBufferCompleted for VkCommandBufferCompleted {
    type CommandBuffer = VkCommandBuffer;

    fn gpu_execution_time(&self) -> Duration {
        self.gpu_execution_time
    }

    fn timestamps(&self) -> &[TimestampSpan] {
        &self.timestamps
    }
}
