use std::{
    mem::forget,
    sync::{Arc, mpsc},
    thread,
};

use ash::vk;
use futures::channel::oneshot;

use crate::backends::vulkan::{Error, VkCommandBufferResources, VkContext, VkSegmentResult};

/// A submitted command buffer, which the context's completion thread waits for, in submission order, to recycle its
/// resources, release what it retained and report its GPU time and span timestamps.
pub struct VkCommandBufferSubmission {
    context: Arc<VkContext>,
    resources: VkCommandBufferResources,
    retained: Vec<Arc<dyn Send + Sync>>,
    sender: oneshot::Sender<VkSegmentResult>,
}

impl VkCommandBufferSubmission {
    /// Starts a completion thread, which exits when its sender is dropped.
    pub fn spawn_completion_thread() -> std::io::Result<mpsc::Sender<Self>> {
        let (sender, receiver) = mpsc::channel::<Self>();
        thread::Builder::new()
            .name("uzu-vulkan-completion".into())
            .spawn(move || receiver.into_iter().for_each(Self::complete))?;
        Ok(sender)
    }

    /// Submits the command buffer of `resources` and returns the receiver of its GPU time and span timestamps, or the
    /// submission error.
    ///
    /// # Safety
    /// `resources` must come from `context` and hold an ended command buffer whose recorded commands reference only
    /// objects kept alive by `retained`.
    pub unsafe fn submit(
        context: Arc<VkContext>,
        resources: VkCommandBufferResources,
        retained: Vec<Arc<dyn Send + Sync>>,
    ) -> Result<oneshot::Receiver<VkSegmentResult>, Error> {
        let command_buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(resources.command_buffer())];
        let submits = [vk::SubmitInfo2::default().command_buffer_infos(&command_buffers)];
        let (sender, receiver) = oneshot::channel();
        // Handing submissions over under the queue lock keeps completions in submission order.
        let queue = context.queue();
        match unsafe { context.device().queue_submit2(*queue, &submits, resources.fence()) } {
            Ok(()) => {
                let submission = Self {
                    context: context.clone(),
                    resources,
                    retained,
                    sender,
                };
                // Without its completion thread the GPU's use of the submission never ends, so nothing is freed and
                // the dropped sender fails the wait.
                if let Err(mpsc::SendError(submission)) = context.completion().send(submission) {
                    forget((submission.context, submission.resources, submission.retained));
                }
            },
            Err(error) => {
                unsafe { context.recycle_command_buffer_resources(resources) };
                return Err(error.into());
            },
        }
        Ok(receiver)
    }

    fn complete(self) {
        let Self {
            context,
            resources,
            retained,
            sender,
        } = self;
        let result = match unsafe { context.device().wait_for_fences(&[resources.fence()], true, u64::MAX) } {
            Ok(()) => {
                let completion = resources
                    .gpu_execution_time(&context)
                    .and_then(|gpu_execution_time| Ok((gpu_execution_time, resources.span_timestamps(&context)?)));
                unsafe { context.recycle_command_buffer_resources(resources) };
                drop(retained);
                // Validation errors fail the next completion, which the work that caused them awaits.
                context.take_validation_error().map_or(completion, |message| Err(Error::Validation(message)))
            },
            // After a failed wait the GPU may still reference the submission, so nothing it uses is freed.
            Err(error) => {
                forget((resources, retained));
                Err(error.into())
            },
        };
        let _ = sender.send(result);
    }
}
