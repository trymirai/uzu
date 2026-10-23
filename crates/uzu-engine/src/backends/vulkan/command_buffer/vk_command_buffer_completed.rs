use std::time::Duration;

use crate::backends::{
    common::{CommandBufferCompleted, TimestampSpan},
    vulkan::VkCommandBuffer,
};

/// A finished submission; its command resources were already recycled.
pub struct VkCommandBufferCompleted {
    gpu_execution_time: Duration,
    timestamps: Box<[TimestampSpan]>,
    timestamp_uncertainty: Option<Duration>,
}

impl VkCommandBufferCompleted {
    pub fn new(
        gpu_execution_time: Duration,
        timestamps: Box<[TimestampSpan]>,
        timestamp_uncertainty: Option<Duration>,
    ) -> Self {
        Self {
            gpu_execution_time,
            timestamps,
            timestamp_uncertainty,
        }
    }

    /// How far each span instant may be from the host clock: the device clock reading's maximum deviation plus half
    /// its host read time. Differences between this command buffer's instants are exact to a device tick. None when
    /// the command buffer recorded no timestamps.
    pub fn timestamp_uncertainty(&self) -> Option<Duration> {
        self.timestamp_uncertainty
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
