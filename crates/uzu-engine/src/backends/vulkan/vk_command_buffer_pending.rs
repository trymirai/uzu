use std::{
    mem::{forget, take},
    sync::Arc,
};

use super::{Error, VkCommandBuffer, VkCommandBufferCompleted, VkCommandBufferResources, VkContext};
use crate::backends::common::CommandBufferPending;

/// A submitted command buffer. Dropping it waits for the GPU before releasing anything it retains.
pub struct VkCommandBufferPending {
    context: Arc<VkContext>,
    resources: Option<VkCommandBufferResources>,
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
        retained: Vec<Arc<dyn Send + Sync>>,
        submit_error: Option<Error>,
    ) -> Self {
        Self {
            context,
            resources: Some(resources),
            retained,
            submitted: submit_error.is_none(),
            submit_error,
        }
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
        forget(take(&mut self.retained));
    }
}

impl CommandBufferPending for VkCommandBufferPending {
    type CommandBuffer = VkCommandBuffer;

    async fn wait_until_completed(mut self) -> Result<VkCommandBufferCompleted, Error> {
        if let Some(error) = self.submit_error.take() {
            return Err(error);
        }
        if let Err(error) = self.wait() {
            self.retain_forever();
            return Err(error);
        }
        self.retained.clear();
        let resources = self.resources.as_ref().expect("pending owns resources until completion");
        Ok(VkCommandBufferCompleted::new(resources.gpu_execution_time(&self.context)?))
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
