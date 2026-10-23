use std::time::Duration;

use ash::vk;

use crate::backends::vulkan::{Error, VkCommandBufferError, VkContext};

/// Timestamps per pool of span timestamps.
const SPAN_TIMESTAMPS_PER_POOL: u32 = 256;

/// Raw per-encoding command pool, primary command buffer, fence, start and end timestamps and the timestamps of
/// recorded spans. They hold no context reference, so the context can cache idle sets without an `Arc` cycle and
/// destroys them before the device.
pub struct VkCommandBufferResources {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    timestamps: vk::QueryPool,
    /// Pools of span timestamps, added as recording needs them and kept for reuse, and the timestamps written.
    span_timestamps: Vec<vk::QueryPool>,
    span_timestamp_count: u32,
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
            unsafe { device.create_command_pool(&pool_info, None) }.map_err(VkCommandBufferError::PoolCreate)?;
        let buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&buffer_info) }
            .inspect_err(|_| unsafe { device.destroy_command_pool(command_pool, None) })?[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .inspect_err(|_| unsafe { device.destroy_command_pool(command_pool, None) })?;
        let timestamps = create_timestamp_pool(device, 2).inspect_err(|_| unsafe {
            device.destroy_fence(fence, None);
            device.destroy_command_pool(command_pool, None);
        })?;
        Ok(Self {
            command_pool,
            command_buffer,
            fence,
            timestamps,
            span_timestamps: Vec::new(),
            span_timestamp_count: 0,
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

    /// Records the next span timestamp, once all earlier commands complete, and returns its index.
    ///
    /// # Safety
    /// The command buffer must be recording on `device`.
    pub unsafe fn write_span_timestamp(
        &mut self,
        device: &ash::Device,
    ) -> Result<u32, Error> {
        let index = self.span_timestamp_count;
        let pool = (index / SPAN_TIMESTAMPS_PER_POOL) as usize;
        if pool == self.span_timestamps.len() {
            self.span_timestamps.push(create_timestamp_pool(device, SPAN_TIMESTAMPS_PER_POOL)?);
        }
        unsafe {
            device.cmd_write_timestamp2(
                self.command_buffer,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.span_timestamps[pool],
                index % SPAN_TIMESTAMPS_PER_POOL,
            )
        };
        self.span_timestamp_count += 1;
        Ok(index)
    }

    /// The time between the two timestamps of a completed command buffer.
    pub fn gpu_execution_time(
        &self,
        context: &VkContext,
    ) -> Result<Duration, Error> {
        let mut results = [[0u64; 2]; 2];
        read_timestamps(context.device(), self.timestamps, &mut results, 0)?;
        Ok(context.timestamp_duration(results[1][0].wrapping_sub(results[0][0])))
    }

    /// The span timestamps of a completed command buffer in device ticks, by index; an unwritten one is an error.
    pub fn span_timestamps(
        &self,
        context: &VkContext,
    ) -> Result<Box<[u64]>, Error> {
        let mut results = vec![[0u64; 2]; self.span_timestamp_count as usize];
        for (index, (pool, results)) in
            self.span_timestamps.iter().zip(results.chunks_mut(SPAN_TIMESTAMPS_PER_POOL as usize)).enumerate()
        {
            read_timestamps(context.device(), *pool, results, 2 + index as u32 * SPAN_TIMESTAMPS_PER_POOL)?;
        }
        Ok(results.into_iter().map(|[ticks, _]| ticks).collect())
    }

    /// Returns the command buffer to the initial state, the fence to unsignaled and the timestamps to unavailable. Only
    /// the span pools the last recording wrote need resetting; the rest stayed reset since their creation or last reset.
    pub fn reset(
        &mut self,
        device: &ash::Device,
    ) -> Result<(), Error> {
        let written_pools = self.span_timestamp_count.div_ceil(SPAN_TIMESTAMPS_PER_POOL) as usize;
        unsafe {
            device.reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())?;
            device.reset_fences(&[self.fence])?;
            device.reset_query_pool(self.timestamps, 0, 2);
            for &pool in &self.span_timestamps[..written_pools] {
                device.reset_query_pool(pool, 0, SPAN_TIMESTAMPS_PER_POOL);
            }
        }
        self.span_timestamp_count = 0;
        Ok(())
    }

    /// # Safety
    /// The resources must belong to `device` and must not be pending on the GPU.
    pub unsafe fn destroy(
        self,
        device: &ash::Device,
    ) {
        unsafe {
            for pool in self.span_timestamps {
                device.destroy_query_pool(pool, None);
            }
            device.destroy_query_pool(self.timestamps, None);
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}

/// Reads the first `results.len()` timestamps of `pool` with their availability, after the submission that writes them
/// completed. One it never wrote is an error naming it: the submission's span timestamps are numbered after its start
/// and end, from `first_query`.
fn read_timestamps(
    device: &ash::Device,
    pool: vk::QueryPool,
    results: &mut [[u64; 2]],
    first_query: u32,
) -> Result<(), Error> {
    let flags = vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY;
    // An unavailable timestamp makes the read NOT_READY, which still writes every availability.
    match unsafe { device.get_query_pool_results(pool, 0, results, flags) } {
        Ok(()) | Err(vk::Result::NOT_READY) => {},
        Err(error) => return Err(error.into()),
    }
    match results.iter().position(|&[_, available]| available == 0) {
        Some(missing) => Err(VkCommandBufferError::UnexecutedSubmission {
            query: first_query + missing as u32,
        }
        .into()),
        None => Ok(()),
    }
}

/// A pool of `count` timestamps, reset so commands can write them.
fn create_timestamp_pool(
    device: &ash::Device,
    count: u32,
) -> Result<vk::QueryPool, Error> {
    let query_info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(count);
    let pool = unsafe { device.create_query_pool(&query_info, None) }?;
    unsafe { device.reset_query_pool(pool, 0, count) };
    Ok(pool)
}
