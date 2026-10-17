use std::time::Duration;

use ash::vk;

use super::{Error, VkContext, VkContextError};

/// Raw per-encoding command pool, primary command buffer, fence and start and end timestamps. They hold no context
/// reference, so the context can cache idle sets without an `Arc` cycle and destroys them before the device.
pub struct VkCommandBufferResources {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    timestamps: vk::QueryPool,
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
        let query_info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2);
        let timestamps = unsafe { device.create_query_pool(&query_info, None) }.inspect_err(|_| unsafe {
            device.destroy_fence(fence, None);
            device.destroy_command_pool(command_pool, None);
        })?;
        unsafe { device.reset_query_pool(timestamps, 0, 2) };
        Ok(Self {
            command_pool,
            command_buffer,
            fence,
            timestamps,
        })
    }

    pub fn command_buffer(&self) -> vk::CommandBuffer {
        self.command_buffer
    }

    pub fn fence(&self) -> vk::Fence {
        self.fence
    }

    /// Records timestamp `query`: 0 at the start of the command buffer, 1 at its end.
    ///
    /// # Safety
    /// The command buffer must be recording on `device`.
    pub unsafe fn write_timestamp(
        &self,
        device: &ash::Device,
        query: u32,
    ) {
        unsafe {
            device.cmd_write_timestamp2(
                self.command_buffer,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.timestamps,
                query,
            )
        };
    }

    /// The time between the two timestamps of a completed command buffer.
    pub fn gpu_execution_time(
        &self,
        context: &VkContext,
    ) -> Result<Duration, Error> {
        let mut results = [0u64; 2];
        unsafe {
            context.device().get_query_pool_results(self.timestamps, 0, &mut results, vk::QueryResultFlags::TYPE_64)
        }?;
        // Timestamps wrap at the queue's valid bits.
        let (ticks, valid_bits) = (results[1].wrapping_sub(results[0]), context.timestamp_valid_bits());
        let ticks = if valid_bits < 64 {
            ticks & ((1u64 << valid_bits) - 1)
        } else {
            ticks
        };
        let period = f64::from(context.physical_device().properties.limits.timestamp_period);
        Ok(Duration::from_nanos((ticks as f64 * period).round() as u64))
    }

    /// Returns the command buffer to the initial state, the fence to unsignaled and the timestamps to unavailable.
    pub fn reset(
        &self,
        device: &ash::Device,
    ) -> Result<(), Error> {
        unsafe {
            device.reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())?;
            device.reset_fences(&[self.fence])?;
            device.reset_query_pool(self.timestamps, 0, 2);
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
            device.destroy_query_pool(self.timestamps, None);
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}
