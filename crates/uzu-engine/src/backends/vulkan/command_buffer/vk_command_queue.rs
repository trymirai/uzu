use std::{
    mem::forget,
    sync::{Arc, atomic::AtomicUsize, mpsc},
    thread,
};

use ash::vk;
use futures::channel::oneshot;
use parking_lot::Mutex;

use crate::backends::vulkan::{Error, VkCommandBufferError, VkContext, VkContextError};

pub struct VkCommandQueue {
    device: ash::Device,
    queue_family_index: u32,
    queue: Mutex<vk::Queue>,
    idle_pools: Mutex<Vec<VkCommandPool>>,
    completion: mpsc::Sender<VkCommandBufferSubmission>,
    pub unsubmitted: AtomicUsize,
}

impl VkCommandQueue {
    pub fn new(
        device: ash::Device,
        queue_family_index: u32,
    ) -> Result<Self, VkContextError> {
        let (completion, submissions) = mpsc::channel::<VkCommandBufferSubmission>();
        thread::Builder::new()
            .name("uzu-vulkan-completion".into())
            .spawn(move || submissions.into_iter().for_each(VkCommandBufferSubmission::complete))
            .map_err(VkContextError::CompletionThreadSpawn)?;
        Ok(Self {
            queue: Mutex::new(unsafe { device.get_device_queue(queue_family_index, 0) }),
            device,
            queue_family_index,
            idle_pools: Mutex::new(Vec::new()),
            completion,
            unsubmitted: AtomicUsize::new(0),
        })
    }

    pub fn acquire(&self) -> Result<VkCommandPool, VkCommandBufferError> {
        let Some(mut command_pool) = self.idle_pools.lock().pop() else {
            return VkCommandPool::new(&self.device, self.queue_family_index);
        };
        if let Err(error) = command_pool.reset(&self.device) {
            unsafe { command_pool.destroy(&self.device) };
            return Err(error);
        }
        Ok(command_pool)
    }

    pub fn recycle(
        &self,
        command_pool: VkCommandPool,
    ) {
        self.idle_pools.lock().push(command_pool);
    }
}

impl Drop for VkCommandQueue {
    fn drop(&mut self) {
        for command_pool in self.idle_pools.get_mut().drain(..) {
            unsafe { command_pool.destroy(&self.device) };
        }
    }
}

pub struct VkCommandBufferSubmission {
    context: Arc<VkContext>,
    command_pool: VkCommandPool,
    retained: Vec<Arc<dyn Send + Sync>>,
    sender: oneshot::Sender<Result<(), Error>>,
}

impl VkCommandBufferSubmission {
    pub unsafe fn submit(
        context: Arc<VkContext>,
        command_pool: VkCommandPool,
        retained: Vec<Arc<dyn Send + Sync>>,
    ) -> Result<oneshot::Receiver<Result<(), Error>>, VkCommandBufferError> {
        let command_buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(command_pool.command_buffer())];
        let submits = [vk::SubmitInfo2::default().command_buffer_infos(&command_buffers)];
        let (sender, receiver) = oneshot::channel();
        let command_queue = context.command_queue();
        let queue = command_queue.queue.lock();
        match unsafe { context.device().queue_submit2(*queue, &submits, command_pool.fence()) } {
            Ok(()) => {
                let submission = Self {
                    context: context.clone(),
                    command_pool,
                    retained,
                    sender,
                };
                if let Err(mpsc::SendError(submission)) = command_queue.completion.send(submission) {
                    forget((submission.context, submission.command_pool, submission.retained));
                }
            },
            Err(error) => {
                command_queue.recycle(command_pool);
                return Err(error.into());
            },
        }
        Ok(receiver)
    }

    fn complete(self) {
        let Self {
            context,
            command_pool,
            retained,
            sender,
        } = self;
        let result = match unsafe { context.device().wait_for_fences(&[command_pool.fence()], true, u64::MAX) } {
            Ok(()) => {
                context.command_queue().recycle(command_pool);
                drop(retained);
                if !sender.is_canceled()
                    && let Some(message) = context.take_validation_error()
                {
                    Err(Error::Validation(message))
                } else {
                    Ok(())
                }
            },
            Err(error) => {
                forget((command_pool, retained));
                Err(VkCommandBufferError::from(error).into())
            },
        };
        drop(context);
        let _ = sender.send(result);
    }
}

// A command pool with its one command buffer and the fence that signals when the GPU has executed it.
pub struct VkCommandPool {
    handle: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl VkCommandPool {
    pub fn new(
        device: &ash::Device,
        queue_family_index: u32,
    ) -> Result<Self, VkCommandBufferError> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let handle = unsafe { device.create_command_pool(&pool_info, None) }?;
        let buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(handle)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&buffer_info) }
            .inspect_err(|_| unsafe { device.destroy_command_pool(handle, None) })?[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .inspect_err(|_| unsafe { device.destroy_command_pool(handle, None) })?;
        Ok(Self {
            handle,
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
            device.reset_command_pool(self.handle, vk::CommandPoolResetFlags::RELEASE_RESOURCES)?;
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
            device.destroy_command_pool(self.handle, None);
        }
    }
}
