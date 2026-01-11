use std::sync::Arc;
use ash::vk;
use crate::backends::vulkan::context::VkContext;

pub struct VkCommandBuffer {
    device: Arc<ash::Device>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
}

impl VkCommandBuffer {
    pub fn new(
        ctx: &VkContext,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let command_pool = {
            let info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.queue_family_index())
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            unsafe { ctx.device().create_command_pool(&info, None)? }
        };
        let command_buffer = {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            unsafe { ctx.device().allocate_command_buffers(&info)? }
        }[0];

        Ok(Self {
            device: ctx.device(),
            command_pool,
            command_buffer,
        })
    }

    pub fn command_buffer(&self) -> vk::CommandBuffer {
        self.command_buffer
    }
}

impl Drop for VkCommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.free_command_buffers(self.command_pool, &[self.command_buffer]);
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}