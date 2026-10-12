use std::sync::Arc;

use ash::vk;

use super::{Error, VkContext};

pub struct VkCommandBuffers {
    context: Arc<VkContext>,
    command_buffers: Vec<vk::CommandBuffer>,
    primary: bool,
}
impl VkCommandBuffers {
    pub fn new(
        ctx: Arc<VkContext>,
        primary: bool,
        count: u32,
    ) -> Result<Self, Error> {
        let level = if primary {
            vk::CommandBufferLevel::PRIMARY
        } else {
            vk::CommandBufferLevel::SECONDARY
        };

        let command_buffers = {
            let command_pool = ctx.command_pool();
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(*command_pool)
                .level(level)
                .command_buffer_count(count);
            unsafe { ctx.device().allocate_command_buffers(&info)? }
        };

        Ok(Self {
            context: ctx,
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
            self.context.device().free_command_buffers(*self.context.command_pool(), self.command_buffers.as_slice());
        }
    }
}
