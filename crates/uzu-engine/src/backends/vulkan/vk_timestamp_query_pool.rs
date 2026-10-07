use std::sync::Arc;

use ash::vk;

use super::{Error, VkContext};

pub struct VkTimestampQueryPool {
    context: Arc<VkContext>,
    timestamp_period: f64,
    query_pool: vk::QueryPool,
    queries_count: u32,
    query_index: u32,
}
impl VkTimestampQueryPool {
    pub fn new(
        ctx: Arc<VkContext>,
        queries_count: u32,
    ) -> Result<Self, Error> {
        let query_pool = {
            let info =
                vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(queries_count);
            unsafe { ctx.device().create_query_pool(&info, None) }?
        };
        Ok(Self {
            timestamp_period: ctx.physical_device().properties.limits.timestamp_period as f64,
            query_pool,
            queries_count,
            query_index: 0,
            context: ctx,
        })
    }

    pub fn write(
        &mut self,
        command_buffer: vk::CommandBuffer,
    ) -> Result<(), Error> {
        if self.query_index == self.queries_count {
            return Err(std::io::Error::other("Timestamp query pool out of range").into());
        }

        if self.query_index == 0 {
            unsafe {
                self.context.device().cmd_reset_query_pool(command_buffer, self.query_pool, 0, self.queries_count);
            }
        }

        unsafe {
            self.context.device().cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                self.query_pool,
                self.query_index,
            );
        }
        self.query_index += 1;
        Ok(())
    }

    pub fn get_duration_nanos(
        &self,
        period_position: u32,
    ) -> Result<f64, Error> {
        if !period_position.checked_add(1).is_some_and(|end| end < self.query_index) {
            return Err(std::io::Error::other("Timestamp query period out of range").into());
        }
        let mut results = vec![0u64; self.query_index as usize].into_boxed_slice();
        unsafe {
            self.context.device().get_query_pool_results(
                self.query_pool,
                0,
                &mut results,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }
        let position = period_position as usize;
        Ok(self.compute_duration_nanos(results[position], results[position + 1]))
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
            self.context.device().destroy_query_pool(self.query_pool, None);
        }
    }
}
