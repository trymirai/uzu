use std::{mem::take, sync::Arc, time::Duration};

use futures::channel::oneshot;

use crate::backends::{
    common::{CommandBuffer, CommandBufferCompleted, CommandBufferExecutable, CommandBufferPending, TimestampSpan},
    vulkan::{Error, VkCommandBufferEncoding, VkCommandBufferError, VkCommandPool, VkContext, Vulkan},
};

pub struct VkCommandBuffer;

impl CommandBuffer for VkCommandBuffer {
    type Backend = Vulkan;

    type Encoding = VkCommandBufferEncoding;
    type Executable = VkCommandBufferExecutable;
    type Pending = VkCommandBufferPending;
    type Completed = VkCommandBufferCompleted;
}

pub struct VkCommandBufferExecutable {
    context: Arc<VkContext>,
    command_pool: Option<VkCommandPool>,
    retained: Vec<Arc<dyn Send + Sync>>,
    segments: Vec<oneshot::Receiver<Result<(), Error>>>,
}

impl VkCommandBufferExecutable {
    pub fn new(
        context: Arc<VkContext>,
        command_pool: VkCommandPool,
        retained: Vec<Arc<dyn Send + Sync>>,
        segments: Vec<oneshot::Receiver<Result<(), Error>>>,
    ) -> Self {
        Self {
            context,
            command_pool: Some(command_pool),
            retained,
            segments,
        }
    }

    pub fn try_submit(mut self) -> Result<VkCommandBufferPending, Error> {
        let command_pool = self.command_pool.take().expect("executable owns its command pool until submission");
        let last = self.context.command_queue().submit(self.context.clone(), command_pool, take(&mut self.retained))?;
        let mut segments = take(&mut self.segments);
        segments.push(last);
        Ok(VkCommandBufferPending {
            segments,
        })
    }
}

impl CommandBufferExecutable for VkCommandBufferExecutable {
    type CommandBuffer = VkCommandBuffer;

    fn submit(self) -> VkCommandBufferPending {
        self.try_submit().unwrap_or_else(|error| VkCommandBufferPending {
            segments: vec![failed_segment(error)],
        })
    }
}

impl Drop for VkCommandBufferExecutable {
    fn drop(&mut self) {
        if let Some(command_pool) = self.command_pool.take() {
            self.context.command_queue().recycle(command_pool);
        }
    }
}

pub struct VkCommandBufferCompleted;

impl CommandBufferCompleted for VkCommandBufferCompleted {
    type CommandBuffer = VkCommandBuffer;

    fn gpu_execution_time(&self) -> Option<Duration> {
        None
    }

    fn timestamps(&self) -> &[TimestampSpan] {
        unimplemented!()
    }
}

pub struct VkCommandBufferPending {
    segments: Vec<oneshot::Receiver<Result<(), Error>>>,
}

impl CommandBufferPending for VkCommandBufferPending {
    type CommandBuffer = VkCommandBuffer;

    async fn wait_until_completed(self) -> Result<VkCommandBufferCompleted, Error> {
        for segment in self.segments {
            segment.await.map_err(|_| VkCommandBufferError::Wait)??;
        }
        Ok(VkCommandBufferCompleted)
    }
}

// A segment that could not be submitted, so waiting for its command buffer reports the error.
pub fn failed_segment(error: Error) -> oneshot::Receiver<Result<(), Error>> {
    let (sender, receiver) = oneshot::channel();
    let _ = sender.send(Err(error));
    receiver
}
