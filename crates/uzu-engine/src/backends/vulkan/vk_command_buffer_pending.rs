use std::{
    mem::{forget, take},
    sync::Arc,
    time::Duration,
};

use super::{Error, VkCommandBufferCompleted, VkCommandBufferResources, VkContext, VkTimestampQueryPool};

/// A submitted command buffer. Dropping it waits for the GPU before releasing anything it retains.
pub struct VkCommandBufferPending {
    context: Arc<VkContext>,
    resources: Option<VkCommandBufferResources>,
    timestamps: Option<VkTimestampQueryPool>,
    retained: Vec<Arc<dyn Send + Sync>>,
    submitted: bool,
    submit_error: Option<Error>,
}

impl VkCommandBufferPending {
    /// # Safety
    /// When `submit_error` is `None`, `resources` must have been submitted with its fence.
    pub unsafe fn new(
        context: Arc<VkContext>,
        resources: VkCommandBufferResources,
        timestamps: VkTimestampQueryPool,
        retained: Vec<Arc<dyn Send + Sync>>,
        submit_error: Option<Error>,
    ) -> Self {
        Self {
            context,
            resources: Some(resources),
            timestamps: Some(timestamps),
            retained,
            submitted: submit_error.is_none(),
            submit_error,
        }
    }

    pub fn wait_until_completed(mut self) -> Result<VkCommandBufferCompleted, Error> {
        if let Some(error) = self.submit_error.take() {
            return Err(error);
        }
        if let Err(error) = self.wait() {
            self.retain_forever();
            return Err(error);
        }
        self.retained.clear();
        let nanos = self.timestamps.take().expect("pending owns timestamps until completion").get_duration_nanos(0)?;
        Ok(VkCommandBufferCompleted::new(Duration::from_nanos(nanos.round() as u64)))
    }

    fn wait(&mut self) -> Result<(), Error> {
        let Some(resources) = self.resources.as_ref().filter(|_| self.submitted) else {
            return Ok(());
        };
        unsafe { self.context.device().wait_for_fences(&[resources.fence()], true, u64::MAX) }?;
        self.submitted = false;
        Ok(())
    }

    /// After a failed wait the GPU may still reference the submission, so nothing it uses is freed.
    fn retain_forever(&mut self) {
        self.submitted = false;
        self.resources = None;
        forget(self.timestamps.take());
        forget(take(&mut self.retained));
    }
}

impl Drop for VkCommandBufferPending {
    fn drop(&mut self) {
        if let Err(error) = self.wait() {
            // Retain first: a panicking logger must not let the GPU-referenced resources be freed.
            self.retain_forever();
            let message = format!("dropped Vulkan command buffer failed to complete, retaining its resources: {error}");
            self.context.logger().e(&message);
        } else if let Some(resources) = self.resources.take() {
            unsafe { self.context.recycle_command_buffer_resources(resources) };
        }
    }
}
