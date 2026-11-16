use std::{
    mem::forget,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
};

use ash::vk;
use futures::channel::oneshot;
use parking_lot::Mutex;

use crate::backends::vulkan::{Error, VkCommandBufferError, VkContext, VkContextError, VkDevice};

pub struct VkCommandQueue {
    device: Arc<VkDevice>,
    queue: Mutex<vk::Queue>,
    idle_pools: Mutex<Vec<VkCommandPool>>,
    completion: mpsc::Sender<VkCommandBufferSubmission>,
    // Command buffers handed out by `acquire` that were neither submitted nor recycled yet.
    unsubmitted: AtomicUsize,
    // Submitted command buffers that the GPU has not finished.
    executing: AtomicUsize,
}

impl VkCommandQueue {
    pub fn new(device: Arc<VkDevice>) -> Result<Self, VkContextError> {
        let (completion, submissions) = mpsc::channel::<VkCommandBufferSubmission>();
        thread::Builder::new()
            .name("uzu-vulkan-completion".into())
            .spawn(move || submissions.into_iter().for_each(VkCommandBufferSubmission::complete))
            .map_err(VkContextError::CompletionThreadSpawn)?;
        Ok(Self {
            queue: Mutex::new(unsafe { device.get_device_queue(device.physical_device.queue_family_index, 0) }),
            device,
            idle_pools: Mutex::new(Vec::new()),
            completion,
            unsubmitted: AtomicUsize::new(0),
            executing: AtomicUsize::new(0),
        })
    }

    // Returns a pool whose command buffer is recording.
    pub fn acquire(&self) -> Result<VkCommandPool, VkCommandBufferError> {
        let idle = self.idle_pools.lock().pop();
        let command_pool = match idle {
            Some(command_pool) => command_pool,
            None => VkCommandPool::new(&self.device, self.device.physical_device.queue_family_index)?,
        };
        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let begun = unsafe {
            self.device
                .reset_command_pool(command_pool.handle, vk::CommandPoolResetFlags::RELEASE_RESOURCES)
                .and_then(|()| self.device.reset_fences(&[command_pool.fence]))
                .and_then(|()| self.device.begin_command_buffer(command_pool.command_buffer, &begin_info))
        };
        if let Err(error) = begun {
            command_pool.destroy(&self.device);
            return Err(error.into());
        }
        self.unsubmitted.fetch_add(1, Ordering::Relaxed);
        Ok(command_pool)
    }

    // Takes back a pool that was not submitted.
    pub fn recycle(
        &self,
        command_pool: VkCommandPool,
    ) {
        self.unsubmitted.fetch_sub(1, Ordering::Relaxed);
        self.idle_pools.lock().push(command_pool);
    }

    // Whether the caller's command buffer is the only one that has not been submitted.
    pub fn has_single_unsubmitted(&self) -> bool {
        self.unsubmitted.load(Ordering::Relaxed) == 1
    }

    // Whether the GPU has finished everything that was submitted.
    pub fn is_idle(&self) -> bool {
        self.executing.load(Ordering::Relaxed) == 0
    }

    // Ends the recording command buffer and submits it; the receiver resolves when the GPU has executed it.
    pub fn submit(
        &self,
        context: Arc<VkContext>,
        command_pool: VkCommandPool,
        retained: Vec<Arc<dyn Send + Sync>>,
    ) -> Result<oneshot::Receiver<Result<(), Error>>, VkCommandBufferError> {
        let command_buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(command_pool.command_buffer)];
        let submits = [vk::SubmitInfo2::default().command_buffer_infos(&command_buffers)];
        let ended = unsafe { self.device.end_command_buffer(command_pool.command_buffer) };
        // Held until the completion thread has the submission, so it waits for fences in queue order.
        let queue = self.queue.lock();
        let submitted = ended.and_then(|()| unsafe { self.device.queue_submit2(*queue, &submits, command_pool.fence) });
        if let Err(error) = submitted {
            self.recycle(command_pool);
            return Err(error.into());
        }
        self.unsubmitted.fetch_sub(1, Ordering::Relaxed);
        self.executing.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = oneshot::channel();
        let submission = VkCommandBufferSubmission {
            context,
            command_pool,
            retained,
            sender,
        };
        // The completion thread only stops by panicking; the submitted work may still use these objects.
        if let Err(mpsc::SendError(submission)) = self.completion.send(submission) {
            forget((submission.context, submission.command_pool, submission.retained));
        }
        Ok(receiver)
    }
}

impl Drop for VkCommandQueue {
    fn drop(&mut self) {
        for command_pool in self.idle_pools.get_mut().drain(..) {
            command_pool.destroy(&self.device);
        }
    }
}

struct VkCommandBufferSubmission {
    context: Arc<VkContext>,
    command_pool: VkCommandPool,
    retained: Vec<Arc<dyn Send + Sync>>,
    sender: oneshot::Sender<Result<(), Error>>,
}

impl VkCommandBufferSubmission {
    fn complete(self) {
        let Self {
            context,
            command_pool,
            retained,
            sender,
        } = self;
        let waited = unsafe { context.device().wait_for_fences(&[command_pool.fence], true, u64::MAX) };
        context.command_queue().executing.fetch_sub(1, Ordering::Relaxed);
        let result = match waited {
            Ok(()) => {
                context.command_queue().idle_pools.lock().push(command_pool);
                drop(retained);
                Ok(())
            },
            Err(error) => {
                // The wait failed, so the device may still use these objects; they are leaked.
                forget((command_pool, retained));
                Err(VkCommandBufferError::from(error).into())
            },
        };
        context.pipelines().save();
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
    fn new(
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

    fn destroy(
        self,
        device: &ash::Device,
    ) {
        unsafe {
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.handle, None);
        }
    }
}
