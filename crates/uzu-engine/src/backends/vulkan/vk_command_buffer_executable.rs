use std::{mem::take, sync::Arc};

use ash::vk;

use super::{VkCommandBufferPending, VkCommandBufferResources, VkContext, VkTimestampQueryPool};

/// An ended command buffer that has not been submitted yet; dropping it recycles its resources.
pub struct VkCommandBufferExecutable {
    context: Arc<VkContext>,
    resources: Option<VkCommandBufferResources>,
    timestamps: Option<VkTimestampQueryPool>,
    retained: Vec<Arc<dyn Send + Sync>>,
}

impl VkCommandBufferExecutable {
    /// # Safety
    /// `resources` must come from `context` and hold an ended command buffer whose recorded
    /// commands reference only `timestamps` and objects kept alive by `retained`.
    pub unsafe fn new(
        context: Arc<VkContext>,
        resources: VkCommandBufferResources,
        timestamps: VkTimestampQueryPool,
        retained: Vec<Arc<dyn Send + Sync>>,
    ) -> Self {
        Self {
            context,
            resources: Some(resources),
            timestamps: Some(timestamps),
            retained,
        }
    }

    /// Submission errors are reported by `VkCommandBufferPending::wait_until_completed`.
    pub fn submit(mut self) -> VkCommandBufferPending {
        let resources = self.resources.take().expect("executable owns resources until submission");
        let command_buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(resources.command_buffer())];
        let submits = [vk::SubmitInfo2::default().command_buffer_infos(&command_buffers)];
        let result = {
            let queue = self.context.queue();
            unsafe { self.context.device().queue_submit2(*queue, &submits, resources.fence()) }
        };
        let timestamps = self.timestamps.take().expect("executable owns timestamps until submission");
        unsafe {
            VkCommandBufferPending::new(
                self.context.clone(),
                resources,
                timestamps,
                take(&mut self.retained),
                result.err().map(Into::into),
            )
        }
    }
}

impl Drop for VkCommandBufferExecutable {
    fn drop(&mut self) {
        if let Some(resources) = self.resources.take() {
            unsafe { self.context.recycle_command_buffer_resources(resources) };
        }
    }
}
