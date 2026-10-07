use ash::vk;

use super::{Error, VkContextError};

/// Raw per-encoding command pool, primary command buffer and fence.
///
/// Holds no reference to its context so the context can cache idle sets without an `Arc` cycle;
/// the context destroys cached sets before the device.
pub struct VkCommandBufferResources {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl VkCommandBufferResources {
    pub fn new(
        device: &ash::Device,
        queue_family_index: u32,
    ) -> Result<Self, Error> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let command_pool =
            unsafe { device.create_command_pool(&pool_info, None) }.map_err(VkContextError::CommandPoolCreate)?;
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

    /// Returns the command buffer to the initial state and the fence to unsignaled.
    pub fn reset(
        &self,
        device: &ash::Device,
    ) -> Result<(), Error> {
        unsafe {
            device.reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())?;
            device.reset_fences(&[self.fence])?;
        }
        Ok(())
    }

    /// # Safety
    /// The resources must belong to `device` and must not be pending on the GPU.
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
