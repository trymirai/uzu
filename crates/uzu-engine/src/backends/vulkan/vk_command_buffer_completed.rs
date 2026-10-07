use std::time::Duration;

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

    pub fn gpu_execution_time(&self) -> Duration {
        self.gpu_execution_time
    }
}
