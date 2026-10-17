use std::{ffi::CString, mem::take, ops::Range, sync::Arc};

use ash::vk;
use rangemap::RangeSet;

use super::{
    Error, VkBuffer, VkCommandBuffer, VkCommandBufferExecutable, VkCommandBufferResources, VkComputePipeline,
    VkContext, VkDebugGroupSpan, Vulkan,
};
use crate::backends::common::{
    Backend, BufferMut, BufferRef, CommandBufferEncoding,
    allocator::{bump::BumpAllocator, pool::PoolAllocator},
};

/// Smallest buffer that holds uploads, so small uploads of one command buffer share it.
const UPLOAD_PAGE_SIZE: usize = 4096;

/// A recording primary command buffer that retains every buffer and pipeline it references and inserts barriers
/// where accesses conflict.
pub struct VkCommandBufferEncoding {
    context: Arc<VkContext>,
    resources: Option<VkCommandBufferResources>,
    retained: Vec<Arc<dyn Send + Sync>>,
    reads: RangeSet<u64>,
    writes: RangeSet<u64>,
    /// Uploads and constants: host-written memory of this command buffer, whose pages dispatches retain.
    uploads: BumpAllocator<Arc<VkBuffer>>,
    /// Scratch pages; a dispatch retains the pages it uses until the command buffer completes.
    allocation_pool: Arc<PoolAllocator<Arc<VkBuffer>>>,
}

impl VkCommandBufferEncoding {
    pub fn new(
        context: Arc<VkContext>,
        allocation_pool: Arc<PoolAllocator<Arc<VkBuffer>>>,
    ) -> Result<Self, Error> {
        let encoding = Self {
            resources: Some(context.acquire_command_buffer_resources()?),
            context,
            retained: Vec::new(),
            reads: RangeSet::new(),
            writes: RangeSet::new(),
            uploads: BumpAllocator::new(UPLOAD_PAGE_SIZE),
            allocation_pool,
        };
        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { encoding.context.device().begin_command_buffer(encoding.command_buffer(), &begin_info)? };
        encoding.memory_barrier(
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_WRITE,
            shader_and_transfer_stages(),
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        );
        unsafe { encoding.resources().write_timestamp(encoding.context.device(), 0) };
        // Execution-only dependency so recorded work cannot run before the start timestamp is written.
        encoding.memory_barrier(
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::NONE,
            shader_and_transfer_stages(),
            vk::AccessFlags2::NONE,
        );
        Ok(encoding)
    }

    pub fn encode_copy(
        &mut self,
        source: &Arc<VkBuffer>,
        source_range: Range<u64>,
        destination: &Arc<VkBuffer>,
        destination_offset: u64,
    ) -> Result<(), Error> {
        let size = source_range.end.saturating_sub(source_range.start);
        let destination_end = destination_offset.checked_add(size).ok_or(Error::BufferRange {
            start: destination_offset,
            end: u64::MAX,
            size: destination.as_ref().size(),
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

    /// Copies `data` into this command buffer's host-coherent memory for dispatches to read; empty data is `None`.
    pub fn upload<T: Copy>(
        &mut self,
        data: &[T],
    ) -> Result<Option<(Arc<VkBuffer>, Range<u64>)>, Error> {
        let size = size_of_val(data);
        if size == 0 {
            return Ok(None);
        }
        let context = self.context.clone();
        let allocation =
            self.uploads.allocate(size, |size| Ok::<_, Error>(Arc::new(VkBuffer::new(context, size as u64)?)))?;
        let (buffer, range) = (allocation.page().clone(), allocation.range());
        assert!(
            (buffer.device_address() + range.start as u64).is_multiple_of(align_of::<T>() as u64),
            "upload of {size} bytes is not aligned to {}",
            align_of::<T>()
        );
        // SAFETY: the range lies in the mapped buffer and belongs to this upload alone, and the GPU
        // cannot be using the buffer: it is not yet part of a submitted command buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr().cast::<u8>(),
                buffer.cpu_ptr().as_ptr().cast::<u8>().add(range.start),
                size,
            );
        }
        Ok(Some((buffer, range.start as u64..range.end as u64)))
    }

    pub fn encode_fill(
        &mut self,
        destination: &Arc<VkBuffer>,
        range: Range<u64>,
        value: u8,
    ) -> Result<(), Error> {
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

    /// Records one compute dispatch after checking the group-count limit and the declared ranges. An empty range is
    /// bounds-checked and retained but not hazard-tracked.
    ///
    /// # Safety
    /// - `push_constants` must be the exact argument block `pipeline`'s shader expects: every device
    ///   address in it points into a buffer listed in `reads` or `writes`, and every scalar is valid.
    /// - `reads` and `writes` must cover every byte the shader may read or write for these `groups`;
    ///   undeclared accesses get no hazard barriers and no lifetime retention.
    /// - `groups` together with the shader's bounds checks must keep every access inside those ranges;
    ///   the shader never dereferences the address of an empty range.
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
        let limits = &self.context.physical_device().properties.limits;
        if groups.iter().zip(limits.max_compute_work_group_count).any(|(&count, limit)| count > limit) {
            return Err(Error::DispatchGroups {
                groups,
                limit: limits.max_compute_work_group_count,
            });
        }
        for (buffer, range) in buffers.clone() {
            if range.start > range.end || range.end > buffer.as_ref().size() {
                return Err(Error::BufferRange {
                    start: range.start,
                    end: range.end,
                    size: buffer.as_ref().size(),
                });
            }
        }
        let address = |(buffer, range): (&Arc<VkBuffer>, Range<u64>)| {
            (!range.is_empty()).then(|| buffer.device_address() + range.start..buffer.device_address() + range.end)
        };
        self.access(reads.filter_map(address), writes.filter_map(address));
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

    pub fn try_end_encoding(mut self) -> Result<VkCommandBufferExecutable, Error> {
        self.memory_barrier(
            shader_and_transfer_stages(),
            vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::HOST,
            vk::AccessFlags2::HOST_READ,
        );
        let device = self.context.device();
        unsafe {
            self.resources().write_timestamp(device, 1);
            device.end_command_buffer(self.command_buffer())?;
        }
        let resources = self.resources.take().expect("encoding owns resources until it ends");
        Ok(unsafe { VkCommandBufferExecutable::new(self.context.clone(), resources, take(&mut self.retained)) })
    }

    pub fn command_buffer(&self) -> vk::CommandBuffer {
        self.resources().command_buffer()
    }

    fn resources(&self) -> &VkCommandBufferResources {
        self.resources.as_ref().expect("encoding owns resources until it ends")
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

impl CommandBufferEncoding for VkCommandBufferEncoding {
    type CommandBuffer = VkCommandBuffer;

    fn context(&self) -> &VkContext {
        &self.context
    }

    fn allocate_constant(
        &mut self,
        size: usize,
    ) -> Result<<Vulkan as Backend>::ConstantBuffer, Error> {
        let context = self.context.clone();
        self.uploads.allocate(size, |size| Ok(Arc::new(VkBuffer::new(context, size as u64)?)))
    }

    fn allocate_scratch(
        &mut self,
        size: usize,
    ) -> Result<<Vulkan as Backend>::ScratchBuffer, Error> {
        let context = self.context.clone();
        self.allocation_pool.allocate(size, |size| Ok(Arc::new(VkBuffer::new(context, size as u64)?)))
    }

    fn encode_copy(
        &mut self,
        src: impl BufferRef<Backend = Vulkan>,
        dst: impl BufferMut<Backend = Vulkan>,
    ) {
        let (source, source_range) = src.parts();
        let (destination, destination_range) = dst.parts();
        assert_eq!(source_range.iter().len(), destination_range.iter().len(), "copy ranges differ in size");
        if source_range.iter().len() == 0 {
            return;
        }
        let (source, source_range) = VkBuffer::resolve(source, source_range);
        let (destination, destination_range) = VkBuffer::resolve(destination, destination_range);
        VkCommandBufferEncoding::encode_copy(self, source, source_range, destination, destination_range.start)
            .expect("Vulkan copy");
    }

    fn encode_fill(
        &mut self,
        dst: impl BufferMut<Backend = Vulkan>,
        value: u8,
    ) {
        let (destination, range) = dst.parts();
        if range.iter().len() == 0 {
            return;
        }
        let (destination, range) = VkBuffer::resolve(destination, range);
        VkCommandBufferEncoding::encode_fill(self, destination, range, value).expect("Vulkan fill");
    }

    fn debug_group(
        &mut self,
        name: &str,
    ) -> VkDebugGroupSpan<'_> {
        VkDebugGroupSpan::new(self, name)
    }

    fn debug_point(
        &mut self,
        name: &str,
    ) {
        if let Some(debug_utils) = self.context.debug_utils() {
            let name = CString::new(name).expect("debug point name contains NUL");
            let label = vk::DebugUtilsLabelEXT::default().label_name(&name);
            unsafe { debug_utils.cmd_insert_debug_utils_label(self.command_buffer(), &label) };
        }
    }

    fn end_encoding(self) -> VkCommandBufferExecutable {
        VkCommandBufferEncoding::try_end_encoding(self).expect("Vulkan command buffer recording")
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
