use std::{
    mem::{forget, take},
    sync::{Arc, atomic::Ordering, mpsc},
    thread,
    time::Duration,
};

use ash::vk;
use futures::channel::oneshot;

use crate::backends::{
    common::{CommandBuffer, CommandBufferCompleted, CommandBufferExecutable, CommandBufferPending, TimestampSpan},
    vulkan::{Error, VkCommandBufferEncoding, VkCommandBufferError, VkContext, Vulkan},
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
    resources: Option<VkCommandBufferResources>,
    retained: Vec<Arc<dyn Send + Sync>>,
    segments: Vec<oneshot::Receiver<Result<(), Error>>>,
    failure: Option<Error>,
}

impl VkCommandBufferExecutable {
    pub unsafe fn new(
        context: Arc<VkContext>,
        resources: VkCommandBufferResources,
        retained: Vec<Arc<dyn Send + Sync>>,
        segments: Vec<oneshot::Receiver<Result<(), Error>>>,
        failure: Option<Error>,
    ) -> Self {
        context.unsubmitted_command_buffers().fetch_add(1, Ordering::Relaxed);
        Self {
            context,
            resources: Some(resources),
            retained,
            segments,
            failure,
        }
    }

    pub fn try_submit(mut self) -> Result<VkCommandBufferPending, Error> {
        if let Some(failure) = self.failure.take() {
            return Err(failure);
        }
        let resources = self.resources.take().expect("executable owns resources until submission");
        let last =
            unsafe { VkCommandBufferSubmission::submit(self.context.clone(), resources, take(&mut self.retained)) }?;
        let mut segments = take(&mut self.segments);
        segments.push(last);
        Ok(VkCommandBufferPending::new(segments))
    }
}

impl CommandBufferExecutable for VkCommandBufferExecutable {
    type CommandBuffer = VkCommandBuffer;

    fn submit(self) -> VkCommandBufferPending {
        self.try_submit().unwrap_or_else(|error| {
            let (sender, receiver) = oneshot::channel();
            let _ = sender.send(Err(error));
            VkCommandBufferPending::new(vec![receiver])
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

impl VkCommandBufferPending {
    pub fn new(segments: Vec<oneshot::Receiver<Result<(), Error>>>) -> Self {
        Self {
            segments,
        }
    }
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

pub struct VkCommandBufferSubmission {
    context: Arc<VkContext>,
    resources: VkCommandBufferResources,
    retained: Vec<Arc<dyn Send + Sync>>,
    sender: oneshot::Sender<Result<(), Error>>,
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
    ) -> Result<oneshot::Receiver<Result<(), Error>>, VkCommandBufferError> {
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
                unsafe { context.recycle_command_buffer_resources(resources) };
                drop(retained);
                if !sender.is_canceled()
                    && let Some(message) = context.take_validation_error()
                {
                    Err(Error::Validation(message))
                } else {
                    context.check_failure().map_err(Error::from)
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

pub struct VkCommandBufferResources {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl VkCommandBufferResources {
    pub fn new(
        device: &ash::Device,
        queue_family_index: u32,
    ) -> Result<Self, VkCommandBufferError> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }?;
        let buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&buffer_info) }
            .inspect_err(|_| unsafe { device.destroy_command_pool(command_pool, None) })?[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .inspect_err(|_| unsafe { device.destroy_command_pool(command_pool, None) })?;
        Ok(Self {
            command_pool,
            command_buffer,
            fence,
        })
    }

    pub fn command_buffer(&self) -> vk::CommandBuffer {
        self.command_buffer
    }

    pub fn fence(&self) -> vk::Fence {
        self.fence
    }

    pub fn reset(
        &mut self,
        device: &ash::Device,
    ) -> Result<(), VkCommandBufferError> {
        unsafe {
            device.reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::RELEASE_RESOURCES)?;
            device.reset_fences(&[self.fence])?;
        }
        Ok(())
    }

    pub unsafe fn destroy(
        self,
        device: &ash::Device,
    ) {
        unsafe {
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}
