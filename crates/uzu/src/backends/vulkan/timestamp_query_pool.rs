use std::sync::Arc;

use ash::vk;

use crate::backends::vulkan::context::VkContext;

pub struct VkTimestampQueryPool {
    device: Arc<ash::Device>,
    timestamp_period: f64,
    query_pool: vk::QueryPool,
    queries_count: u32,
    query_index: u32,
}

impl VkTimestampQueryPool {
    pub fn new(
        ctx: &VkContext,
        queries_count: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let query_pool = {
            let info =
                vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(queries_count);
            unsafe { ctx.device().create_query_pool(&info, None) }?
        };
        Ok(Self {
            device: ctx.device(),
            timestamp_period: ctx.physical_device().properties.limits.timestamp_period as f64,
            query_pool,
            queries_count,
            query_index: 0,
        })
    }

    pub fn write(
        &mut self,
        command_buffer: vk::CommandBuffer,
    ) {
        if self.query_index == self.queries_count {
            eprintln!("Timestamp query pool out of range");
            return;
        }

        if self.query_index == 0 {
            unsafe {
                self.device.cmd_reset_query_pool(command_buffer, self.query_pool, 0, self.queries_count);
            }
        }

        unsafe {
            self.device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                self.query_pool,
                self.query_index,
            );
        }
        self.query_index += 1
    }

    pub fn get_duration_nanos(
        &self,
        period_position: u32,
    ) -> f64 {
        if period_position >= self.queries_count - 1 {
            eprintln!("Timestamp query pool out of range");
            return -1.0f64;
        }

        let mut results = vec![0u64, self.queries_count as u64];
        let result = unsafe {
            self.device.get_query_pool_results(
                self.query_pool,
                0,
                &mut results,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        };
        if let Err(res) = result {
            eprintln!("Can not query pool results: {res:?}");
            return -1.0f64;
        }

        let position = period_position as usize;
        self.compute_duration_nanos(results[position], results[position + 1])
    }

    fn compute_duration_nanos(
        &self,
        start: u64,
        finish: u64,
    ) -> f64 {
        (finish - start) as f64 * self.timestamp_period
    }
}

impl Drop for VkTimestampQueryPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_query_pool(self.query_pool, None);
        }
    }
}
