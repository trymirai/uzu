use std::{
    mem::forget,
    sync::{Arc, mpsc},
    thread,
};

use ash::vk;
use futures::channel::oneshot;

use crate::backends::vulkan::{Error, VkCommandBufferError, VkCommandBufferResources, VkContext, VkSegmentResult};

pub struct VkCommandBufferSubmission {
    context: Arc<VkContext>,
    resources: VkCommandBufferResources,
    retained: Vec<Arc<dyn Send + Sync>>,
    sender: oneshot::Sender<VkSegmentResult>,
}

impl VkCommandBufferSubmission {
    pub fn spawn_completion_thread() -> std::io::Result<mpsc::Sender<Self>> {
        let (sender, receiver) = mpsc::channel::<Self>();
        thread::Builder::new()
            .name("uzu-vulkan-completion".into())
            .spawn(move || receiver.into_iter().for_each(Self::complete))?;
        Ok(sender)
    }

    pub unsafe fn submit(
        context: Arc<VkContext>,
        resources: VkCommandBufferResources,
        retained: Vec<Arc<dyn Send + Sync>>,
    ) -> Result<oneshot::Receiver<VkSegmentResult>, VkCommandBufferError> {
        let command_buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(resources.command_buffer())];
        let submits = [vk::SubmitInfo2::default().command_buffer_infos(&command_buffers)];
        let (sender, receiver) = oneshot::channel();
        let queue = context.queue();
        match unsafe { context.device().queue_submit2(*queue, &submits, resources.fence()) } {
            Ok(()) => {
                let submission = Self {
                    context: context.clone(),
                    resources,
                    retained,
                    sender,
                };
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
                match (!sender.is_canceled()).then(|| context.take_validation_error()).flatten() {
                    Some(message) => Err(Error::Validation(message)),
                    None => context.check_failure().map_err(Error::from).and(completion.map_err(Error::from)),
                }
            },
            Err(error) => {
                forget((resources, retained));
                Err(VkCommandBufferError::from(error).into())
            },
        };
        let weak_context = Arc::downgrade(&context);
        drop(context);
        if let Err(Err(error)) = sender.send(result)
            && let Some(context) = weak_context.upgrade()
        {
            context.fail(error);
        }
    }
}
