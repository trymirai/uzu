use std::{sync::Arc, time::Duration};

use futures::channel::oneshot;

use crate::backends::{
    common::CommandBufferPending,
    vulkan::{
        Error, VkCommandBuffer, VkCommandBufferCompleted, VkCommandBufferError, VkContext, VkSegmentResult,
        VkTimestampRecorder,
    },
};

pub struct VkCommandBufferPending {
    segments: Vec<oneshot::Receiver<VkSegmentResult>>,
    timestamps: Option<(Arc<VkContext>, VkTimestampRecorder)>,
}

impl VkCommandBufferPending {
    pub fn new(
        segments: Vec<oneshot::Receiver<VkSegmentResult>>,
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
