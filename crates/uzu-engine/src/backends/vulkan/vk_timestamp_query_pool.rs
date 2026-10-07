use std::sync::Arc;

use ash::vk;

use super::{Error, VkContext};

pub struct VkTimestampQueryPool {
    context: Arc<VkContext>,
    timestamp_period: f64,
    valid_bits: u32,
    query_pool: vk::QueryPool,
    queries_count: u32,
    query_index: u32,
}
impl VkTimestampQueryPool {
    pub fn new(
        ctx: Arc<VkContext>,
        queries_count: u32,
    ) -> Result<Self, Error> {
        if queries_count == 0 {
            return Err(Error::TimestampRange);
        }
        let valid_bits = ctx.timestamp_valid_bits();
        if valid_bits == 0 {
            return Err(Error::TimestampUnsupported);
        }
        let query_pool = {
            let info =
                vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(queries_count);
            unsafe { ctx.device().create_query_pool(&info, None) }?
        };
        unsafe {
            ctx.device().reset_query_pool(query_pool, 0, queries_count);
        }
        Ok(Self {
            valid_bits,
            timestamp_period: ctx.physical_device().properties.limits.timestamp_period as f64,
            query_pool,
            queries_count,
            query_index: 0,
            context: ctx,
        })
    }

    /// # Safety
    /// The command buffer must be recording on this context, externally synchronized,
    /// and complete before this pool is reset or dropped.
    pub unsafe fn write(
        &mut self,
        command_buffer: vk::CommandBuffer,
    ) -> Result<(), Error> {
        if self.query_index == self.queries_count {
            return Err(Error::TimestampCapacity);
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
            return Err(Error::TimestampRange);
        }
        let mut results = [0u64; 2];
        unsafe {
            self.context.device().get_query_pool_results(
                self.query_pool,
                period_position,
                &mut results,
                vk::QueryResultFlags::TYPE_64,
            )?;
        }
        Ok(Self::compute_duration_nanos(results[0], results[1], self.timestamp_period, self.valid_bits))
    }

    /// # Safety
    /// All commands using this pool must have completed before resetting it.
    pub unsafe fn reset(&mut self) {
        unsafe {
            self.context.device().reset_query_pool(self.query_pool, 0, self.queries_count);
        }
        self.query_index = 0;
    }

    fn compute_duration_nanos(
        start: u64,
        finish: u64,
        timestamp_period: f64,
        valid_bits: u32,
    ) -> f64 {
        let ticks = finish.wrapping_sub(start);
        let ticks = if valid_bits < 64 {
            ticks & ((1u64 << valid_bits) - 1)
        } else {
            ticks
        };
        ticks as f64 * timestamp_period
    }
}
impl Drop for VkTimestampQueryPool {
    fn drop(&mut self) {
        unsafe {
            self.context.device().destroy_query_pool(self.query_pool, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::VkTimestampQueryPool;

    #[uzu_engine_macros::uzu_test]
    fn timestamp_wrap_uses_queue_precision() {
        assert_eq!(VkTimestampQueryPool::compute_duration_nanos(254, 2, 2.0, 8), 8.0);
        assert_eq!(VkTimestampQueryPool::compute_duration_nanos(u64::MAX - 1, 2, 2.0, 64), 8.0);
    }
}
