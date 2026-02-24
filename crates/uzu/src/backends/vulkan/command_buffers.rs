use std::sync::Arc;

use ash::vk;

use crate::backends::vulkan::context::VkContext;

pub struct VkCommandBuffers {
    device: Arc<ash::Device>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    primary: bool,
}

impl VkCommandBuffers {
    pub fn new(
        ctx: &VkContext,
        primary: bool,
        count: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let command_pool = ctx.command_pool();
        let level = if primary {
            vk::CommandBufferLevel::PRIMARY
        } else {
            vk::CommandBufferLevel::SECONDARY
        };

        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(level)
            .command_buffer_count(count);
        let command_buffers = unsafe { ctx.device().allocate_command_buffers(&info)? };

        Ok(Self {
            device: ctx.device(),
            command_pool,
            command_buffers,
            primary,
        })
    }

    pub fn command_buffers(&self) -> &[vk::CommandBuffer] {
        self.command_buffers.as_slice()
    }

    pub fn primary(&self) -> bool {
        self.primary
    }
}

impl Drop for VkCommandBuffers {
    fn drop(&mut self) {
        unsafe {
            self.device.free_command_buffers(self.command_pool, self.command_buffers.as_slice());
        }
    }
}
