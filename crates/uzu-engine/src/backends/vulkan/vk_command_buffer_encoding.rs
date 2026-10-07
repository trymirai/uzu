use std::{mem::take, ops::Range, sync::Arc};

use ash::vk;
use rangemap::RangeSet;

use super::{
    Error, VkBuffer, VkCommandBufferExecutable, VkCommandBufferResources, VkComputePipeline, VkContext,
    VkTimestampQueryPool,
};

/// A recording primary command buffer that owns its pool, retains every buffer and pipeline it
/// references, and inserts barriers where accesses conflict (same algorithm as the Metal encoder).
pub struct VkCommandBufferEncoding {
    context: Arc<VkContext>,
    resources: Option<VkCommandBufferResources>,
    timestamps: Option<VkTimestampQueryPool>,
    retained: Vec<Arc<dyn Send + Sync>>,
    reads: RangeSet<u64>,
    writes: RangeSet<u64>,
}

impl VkCommandBufferEncoding {
    pub fn new(context: Arc<VkContext>) -> Result<Self, Error> {
        let mut encoding = Self {
            resources: Some(context.acquire_command_buffer_resources()?),
            context: context.clone(),
            timestamps: None,
            retained: Vec::new(),
            reads: RangeSet::new(),
            writes: RangeSet::new(),
        };
        let mut timestamps = VkTimestampQueryPool::new(context, 2)?;
        let command_buffer = encoding.command_buffer();
        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { encoding.context.device().begin_command_buffer(command_buffer, &begin_info)? };
        encoding.memory_barrier(
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_WRITE,
            shader_and_transfer_stages(),
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        );
        unsafe { timestamps.write(command_buffer)? };
        // Execution-only dependency so recorded work cannot run before the start timestamp is written.
        encoding.memory_barrier(
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::NONE,
            shader_and_transfer_stages(),
            vk::AccessFlags2::NONE,
        );
        encoding.timestamps = Some(timestamps);
        Ok(encoding)
    }

    pub fn context(&self) -> &Arc<VkContext> {
        &self.context
    }

    pub fn encode_copy(
        &mut self,
        source: &Arc<VkBuffer>,
        source_range: Range<u64>,
        destination: &Arc<VkBuffer>,
        destination_offset: u64,
    ) -> Result<(), Error> {
        self.check_owned([source.context(), destination.context()])?;
        let size = source_range.end.saturating_sub(source_range.start);
        let destination_end = destination_offset.checked_add(size).ok_or(Error::BufferRange {
            start: destination_offset,
            end: u64::MAX,
            size: destination.size(),
        })?;
        let destination_range = destination_offset..destination_end;
        let source_addresses = device_addresses(source, &source_range)?;
        let destination_addresses = device_addresses(destination, &destination_range)?;
        if source_addresses.start < destination_addresses.end && destination_addresses.start < source_addresses.end {
            return Err(Error::CopyOverlap);
        }
        self.access([source_addresses].into_iter(), [destination_addresses].into_iter());
        let region = vk::BufferCopy {
            src_offset: source_range.start,
            dst_offset: destination_offset,
            size,
        };
        unsafe {
            self.context.device().cmd_copy_buffer(
                self.command_buffer(),
                source.buffer(),
                destination.buffer(),
                &[region],
            );
        }
        self.retained.extend([source.clone() as Arc<dyn Send + Sync>, destination.clone()]);
        Ok(())
    }

    pub fn encode_fill(
        &mut self,
        destination: &Arc<VkBuffer>,
        range: Range<u64>,
        value: u8,
    ) -> Result<(), Error> {
        self.check_owned([destination.context()])?;
        let addresses = device_addresses(destination, &range)?;
        if !range.start.is_multiple_of(4) || !range.end.is_multiple_of(4) {
            return Err(Error::FillAlignment {
                start: range.start,
                end: range.end,
            });
        }
        self.access(std::iter::empty(), [addresses].into_iter());
        unsafe {
            self.context.device().cmd_fill_buffer(
                self.command_buffer(),
                destination.buffer(),
                range.start,
                range.end - range.start,
                u32::from_ne_bytes([value; 4]),
            );
        }
        self.retained.push(destination.clone());
        Ok(())
    }

    /// Records one compute dispatch. Ownership, the push-constant size, the group-count limit and
    /// the declared ranges are checked; nothing is recorded when a check fails.
    ///
    /// # Safety
    /// - `push_constants` must be the exact argument block `pipeline`'s shader expects: every device
    ///   address in it points into a buffer listed in `reads` or `writes`, and every scalar is valid.
    /// - `reads` and `writes` must cover every byte the shader may read or write for these `groups`;
    ///   undeclared accesses get no hazard barriers and no lifetime retention.
    /// - `groups` together with the shader's bounds checks must keep every access inside those ranges.
    pub unsafe fn encode_dispatch<'b>(
        &mut self,
        pipeline: &Arc<VkComputePipeline>,
        push_constants: &[u8],
        groups: [u32; 3],
        reads: impl IntoIterator<Item = (&'b Arc<VkBuffer>, Range<u64>), IntoIter: Clone>,
        writes: impl IntoIterator<Item = (&'b Arc<VkBuffer>, Range<u64>), IntoIter: Clone>,
    ) -> Result<(), Error> {
        let (reads, writes) = (reads.into_iter(), writes.into_iter());
        let buffers = reads.clone().chain(writes.clone());
        self.check_owned(
            std::iter::once(pipeline.context()).chain(buffers.clone().map(|(buffer, _)| buffer.context())),
        )?;
        if push_constants.len() != pipeline.push_constant_size() as usize {
            return Err(Error::PushConstantsMismatch {
                size: push_constants.len(),
                expected: pipeline.push_constant_size(),
            });
        }
        let limits = &self.context.physical_device().properties.limits;
        if groups.iter().zip(limits.max_compute_work_group_count).any(|(&count, limit)| count > limit) {
            return Err(Error::DispatchGroups {
                groups,
                limit: limits.max_compute_work_group_count,
            });
        }
        buffers.clone().try_for_each(|(buffer, range)| device_addresses(buffer, &range).map(drop))?;
        let address = |(buffer, range): (&Arc<VkBuffer>, Range<u64>)| {
            buffer.device_address() + range.start..buffer.device_address() + range.end
        };
        self.access(reads.map(address), writes.map(address));
        let command_buffer = self.command_buffer();
        let device = self.context.device();
        unsafe {
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline());
            if !push_constants.is_empty() {
                device.cmd_push_constants(
                    command_buffer,
                    pipeline.pipeline_layout(),
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }
            device.cmd_dispatch(command_buffer, groups[0], groups[1], groups[2]);
        }
        self.retained.push(pipeline.clone());
        self.retained.extend(buffers.map(|(buffer, _)| buffer.clone() as Arc<dyn Send + Sync>));
        Ok(())
    }

    pub fn end_encoding(mut self) -> Result<VkCommandBufferExecutable, Error> {
        self.memory_barrier(
            shader_and_transfer_stages(),
            vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::HOST,
            vk::AccessFlags2::HOST_READ,
        );
        let command_buffer = self.command_buffer();
        let mut timestamps = self.timestamps.take().expect("encoding owns timestamps until it ends");
        unsafe {
            timestamps.write(command_buffer)?;
            self.context.device().end_command_buffer(command_buffer)?;
        }
        let resources = self.resources.take().expect("encoding owns resources until it ends");
        Ok(unsafe {
            VkCommandBufferExecutable::new(self.context.clone(), resources, timestamps, take(&mut self.retained))
        })
    }

    fn check_owned<'a>(
        &self,
        contexts: impl IntoIterator<Item = &'a Arc<VkContext>>,
    ) -> Result<(), Error> {
        match contexts.into_iter().all(|context| Arc::ptr_eq(context, &self.context)) {
            true => Ok(()),
            false => Err(Error::ForeignContext),
        }
    }

    fn command_buffer(&self) -> vk::CommandBuffer {
        self.resources.as_ref().expect("encoding owns resources until it ends").command_buffer()
    }

    fn access(
        &mut self,
        reads: impl Iterator<Item = Range<u64>> + Clone,
        writes: impl Iterator<Item = Range<u64>> + Clone,
    ) {
        if reads.clone().chain(writes.clone()).any(|range| self.writes.overlaps(&range))
            || writes.clone().any(|range| self.reads.overlaps(&range))
        {
            self.memory_barrier(
                shader_and_transfer_stages(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::TRANSFER_WRITE,
                shader_and_transfer_stages(),
                vk::AccessFlags2::SHADER_STORAGE_READ
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE
                    | vk::AccessFlags2::TRANSFER_READ
                    | vk::AccessFlags2::TRANSFER_WRITE,
            );
            self.reads.clear();
            self.writes.clear();
        }
        reads.for_each(|range| self.reads.insert(range));
        writes.for_each(|range| self.writes.insert(range));
    }

    fn memory_barrier(
        &self,
        src_stage_mask: vk::PipelineStageFlags2,
        src_access_mask: vk::AccessFlags2,
        dst_stage_mask: vk::PipelineStageFlags2,
        dst_access_mask: vk::AccessFlags2,
    ) {
        let barriers = [vk::MemoryBarrier2::default()
            .src_stage_mask(src_stage_mask)
            .src_access_mask(src_access_mask)
            .dst_stage_mask(dst_stage_mask)
            .dst_access_mask(dst_access_mask)];
        let dependency = vk::DependencyInfo::default().memory_barriers(&barriers);
        unsafe { self.context.device().cmd_pipeline_barrier2(self.command_buffer(), &dependency) };
    }
}

impl Drop for VkCommandBufferEncoding {
    fn drop(&mut self) {
        if let Some(resources) = self.resources.take() {
            // Never submitted, so the GPU cannot be using it; the pool reset on reuse discards the recording.
            unsafe { self.context.recycle_command_buffer_resources(resources) };
        }
    }
}

fn shader_and_transfer_stages() -> vk::PipelineStageFlags2 {
    vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::ALL_TRANSFER
}

fn device_addresses(
    buffer: &VkBuffer,
    range: &Range<u64>,
) -> Result<Range<u64>, Error> {
    if range.start >= range.end || range.end > buffer.size() {
        return Err(Error::BufferRange {
            start: range.start,
            end: range.end,
            size: buffer.size(),
        });
    }
    Ok(buffer.device_address() + range.start..buffer.device_address() + range.end)
}
