use std::{
    mem::take,
    sync::{Arc, atomic::Ordering},
};

use futures::channel::oneshot;

use crate::backends::{
    common::CommandBufferExecutable,
    vulkan::{
        Error, VkCommandBuffer, VkCommandBufferPending, VkCommandBufferResources, VkCommandBufferSubmission, VkContext,
        VkSegmentResult, VkTimestampRecorder,
    },
};

/// An ended command buffer whose last segment has not been submitted yet; dropping it recycles that segment's
/// resources, while the segments submitted during recording still run.
pub struct VkCommandBufferExecutable {
    context: Arc<VkContext>,
    resources: Option<VkCommandBufferResources>,
    retained: Vec<Arc<dyn Send + Sync>>,
    segments: Vec<oneshot::Receiver<VkSegmentResult>>,
    timestamps: Option<VkTimestampRecorder>,
    /// The error that ended the recording, reported instead of submitting the last segment.
    failure: Option<Error>,
}

impl VkCommandBufferExecutable {
    /// # Safety
    /// `resources` must come from `context` and hold an ended command buffer whose recorded
    /// commands reference only objects kept alive by `retained`, after the submitted `segments`, unless `failure`
    /// ended the recording.
    pub unsafe fn new(
        context: Arc<VkContext>,
        resources: VkCommandBufferResources,
        retained: Vec<Arc<dyn Send + Sync>>,
        segments: Vec<oneshot::Receiver<VkSegmentResult>>,
        timestamps: Option<VkTimestampRecorder>,
        failure: Option<Error>,
    ) -> Self {
        context.unsubmitted_command_buffers().fetch_add(1, Ordering::Relaxed);
        Self {
            context,
            resources: Some(resources),
            retained,
            segments,
            timestamps,
            failure,
        }
    }

    /// Submits the last segment, or returns the error that ended the recording or failed the submission.
    pub fn try_submit(mut self) -> Result<VkCommandBufferPending, Error> {
        if let Some(failure) = self.failure.take() {
            return Err(failure);
        }
        let resources = self.resources.take().expect("executable owns resources until submission");
        let last =
            unsafe { VkCommandBufferSubmission::submit(self.context.clone(), resources, take(&mut self.retained)) }?;
        let mut segments = take(&mut self.segments);
        segments.push(last);
        let timestamps = self.timestamps.take().map(|recorder| (self.context.clone(), recorder));
        Ok(VkCommandBufferPending::new(segments, timestamps))
    }
}

impl CommandBufferExecutable for VkCommandBufferExecutable {
    type CommandBuffer = VkCommandBuffer;

    /// Submission errors are reported by `VkCommandBufferPending::wait_until_completed`.
    fn submit(self) -> VkCommandBufferPending {
        self.try_submit().unwrap_or_else(|error| {
            let (sender, receiver) = oneshot::channel();
            let _ = sender.send(Err(error));
            VkCommandBufferPending::new(vec![receiver], None)
        })
    }
}

impl Drop for VkCommandBufferExecutable {
    fn drop(&mut self) {
        if let Some(resources) = self.resources.take() {
            unsafe { self.context.recycle_command_buffer_resources(resources) };
        }
        self.context.unsubmitted_command_buffers().fetch_sub(1, Ordering::Relaxed);
    }
}
