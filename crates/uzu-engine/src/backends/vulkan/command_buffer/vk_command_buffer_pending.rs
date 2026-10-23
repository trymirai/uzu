use std::{sync::Arc, time::Duration};

use futures::channel::oneshot;

use crate::backends::{
    common::CommandBufferPending,
    vulkan::{Error, VkCommandBuffer, VkCommandBufferCompleted, VkCommandBufferError, VkContext, VkTimestampRecorder},
};

/// A submitted command buffer; the context's completion thread reports when the GPU finishes each of its segments.
pub struct VkCommandBufferPending {
    segments: Vec<oneshot::Receiver<Result<(Duration, Box<[u64]>), Error>>>,
    timestamps: Option<(Arc<VkContext>, VkTimestampRecorder)>,
}

impl VkCommandBufferPending {
    pub fn new(
        segments: Vec<oneshot::Receiver<Result<(Duration, Box<[u64]>), Error>>>,
        timestamps: Option<(Arc<VkContext>, VkTimestampRecorder)>,
    ) -> Self {
        Self {
            segments,
            timestamps,
        }
    }
}

impl CommandBufferPending for VkCommandBufferPending {
    type CommandBuffer = VkCommandBuffer;

    /// The GPU time is the sum of the segments'; spans resolve once every segment completed. Span timestamps are kept
    /// only when recording them, so the default path allocates nothing for them.
    async fn wait_until_completed(self) -> Result<VkCommandBufferCompleted, Error> {
        let mut gpu_execution_time = Duration::ZERO;
        let mut segment_timestamps = Vec::new();
        for segment in self.segments {
            let (segment_time, timestamps) = segment.await.map_err(|_| VkCommandBufferError::Wait)??;
            gpu_execution_time += segment_time;
            if self.timestamps.is_some() {
                segment_timestamps.push(timestamps);
            }
        }
        let timestamps = match &self.timestamps {
            Some((context, recorder)) => recorder.resolve(context, &segment_timestamps)?,
            None => Box::default(),
        };
        Ok(VkCommandBufferCompleted::new(gpu_execution_time, timestamps))
    }
}
