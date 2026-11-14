use std::{
    ffi::CString,
    mem::take,
    ops::{Deref, DerefMut, Range},
    sync::{Arc, atomic::Ordering},
};

use ash::{ext, vk};
use bytemuck::NoUninit;
use futures::channel::oneshot;
use rangemap::RangeSet;

use crate::backends::{
    common::{
        Backend, BufferMut, BufferRef, CommandBufferEncoding,
        allocator::{bump::BumpAllocator, pool::PoolAllocator},
    },
    vulkan::{
        Error, VkBuffer, VkBufferCpuGpu, VkBufferMemory, VkCommandBuffer, VkCommandBufferError,
        VkCommandBufferExecutable, VkCommandBufferResources, VkCommandBufferSubmission, VkComputePipeline, VkContext,
        Vulkan,
    },
};

const UPLOAD_PAGE_SIZE: usize = 4096;
// Declared FLOPs per submission. Windows resets a GPU that works on one submission for 2 seconds:
// https://learn.microsoft.com/en-us/windows-hardware/drivers/display/timeout-detection-and-recovery
const SEGMENT_WORK: u64 = 64_000_000_000;

pub struct VkCommandBufferEncoding {
    context: Arc<VkContext>,
    resources: Option<VkCommandBufferResources>,
    retained: Vec<Arc<dyn Send + Sync>>,
    reads: RangeSet<u64>,
    writes: RangeSet<u64>,
    uploads: BumpAllocator<VkBufferCpuGpu>,
    allocation_pool: Arc<PoolAllocator<Arc<VkBuffer>>>,
    segments: Vec<oneshot::Receiver<Result<(), Error>>>,
    segment_work: u64,
    declared_work: u64,
    named: bool,
    failure: Option<Error>,
}

impl VkCommandBufferEncoding {
    pub fn new(
        context: Arc<VkContext>,
        allocation_pool: Arc<PoolAllocator<Arc<VkBuffer>>>,
        name: Option<&str>,
    ) -> Result<Self, Error> {
        let resources = context.submitter().acquire()?;
        context.submitter().unsubmitted.fetch_add(1, Ordering::Relaxed);
        let encoding = Self {
            resources: Some(resources),
            context,
            retained: Vec::new(),
            reads: RangeSet::new(),
            writes: RangeSet::new(),
            uploads: BumpAllocator::new(UPLOAD_PAGE_SIZE, 1),
            allocation_pool,
            segments: Vec::new(),
            segment_work: 0,
            declared_work: 0,
            named: name.is_some(),
            failure: None,
        };
        encoding.begin_segment()?;
        if let Some(name) = name {
            encoding.begin_debug_label(name);
        }
        Ok(encoding)
    }

    pub fn upload<T: NoUninit>(
        &mut self,
        data: &[T],
    ) -> Option<(Arc<VkBuffer>, Range<u64>)> {
        let size = size_of_val(data);
        if size == 0 || self.failure.is_some() {
            return None;
        }
        let allocation = self
            .allocate_constant_from_slice(bytemuck::cast_slice::<T, u8>(data))
            .map_err(|error| self.failure = Some(error))
            .ok()?;
        let buffer = allocation.page().buffer().clone();
        let range = allocation.range();
        assert!(
            (buffer.device_address() + range.start as u64).is_multiple_of(align_of::<T>() as u64),
            "upload of {size} bytes is not aligned to {}",
            align_of::<T>()
        );
        Some((buffer, range.start as u64..range.end as u64))
    }

    pub unsafe fn encode_dispatch<'b>(
        &mut self,
        kernel: &'static str,
        pipeline: &Arc<VkComputePipeline>,
        push_constants: &[u8],
        groups: [u32; 3],
        reads: impl IntoIterator<Item = (&'b Arc<VkBuffer>, Range<u64>), IntoIter: Clone>,
        writes: impl IntoIterator<Item = (&'b Arc<VkBuffer>, Range<u64>), IntoIter: Clone>,
    ) {
        let reads = reads.into_iter();
        let writes = writes.into_iter();
        let buffers = reads.clone().chain(writes.clone());
        let limits = &self.context.physical_device().limits;
        let checked = if groups.iter().zip(limits.max_work_group_count).any(|(&count, limit)| count > limit) {
            Err(VkCommandBufferError::DispatchGroups {
                groups,
                limit: limits.max_work_group_count,
            })
        } else {
            buffers.clone().try_for_each(|(buffer, range)| device_addresses(buffer, &range).map(drop))
        };
        if let Err(error) = checked {
            self.failure.get_or_insert(VkCommandBufferError::Dispatch(kernel, Box::new(error)).into());
            return;
        }
        let addresses = |(buffer, range): (&Arc<VkBuffer>, Range<u64>)| device_addresses(buffer, &range).ok().flatten();
        if !self.access(reads.filter_map(addresses), writes.filter_map(addresses)) {
            return;
        }
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
    }

    fn command_buffer(&self) -> vk::CommandBuffer {
        self.resources.as_ref().expect("encoding owns resources until it ends").command_buffer()
    }

    fn debug_utils(&self) -> Option<&ext::debug_utils::Device> {
        self.context.debug_labels().filter(|_| self.failure.is_none())
    }

    pub fn begin_debug_label(
        &self,
        name: &str,
    ) {
        if let Some(debug_utils) = self.debug_utils() {
            let name = CString::new(name).expect("debug label contains NUL");
            let label = vk::DebugUtilsLabelEXT::default().label_name(&name);
            unsafe { debug_utils.cmd_begin_debug_utils_label(self.command_buffer(), &label) };
        }
    }

    pub fn end_debug_label(&self) {
        if let Some(debug_utils) = self.debug_utils() {
            unsafe { debug_utils.cmd_end_debug_utils_label(self.command_buffer()) };
        }
    }

    pub fn declare_work(
        &mut self,
        flops: u64,
    ) {
        self.declared_work += flops;
    }

    fn access(
        &mut self,
        reads: impl Iterator<Item = Range<u64>> + Clone,
        writes: impl Iterator<Item = Range<u64>> + Clone,
    ) -> bool {
        if self.failure.is_some() {
            return false;
        }
        let work = take(&mut self.declared_work);
        if self.segment_work > 0
            && self.segment_work + work > SEGMENT_WORK
            && self.context.submitter().unsubmitted.load(Ordering::Relaxed) == 1
        {
            if let Err(error) = self.submit_segment() {
                self.failure = Some(error);
                return false;
            }
        } else if reads.clone().chain(writes.clone()).any(|range| self.writes.overlaps(&range))
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
        self.segment_work += work;
        true
    }

    fn begin_segment(&self) -> Result<(), VkCommandBufferError> {
        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.context.device().begin_command_buffer(self.command_buffer(), &begin_info)? };
        self.memory_barrier(
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_WRITE,
            shader_and_transfer_stages(),
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        );
        Ok(())
    }

    fn end_segment(&self) -> Result<(), VkCommandBufferError> {
        self.memory_barrier(
            shader_and_transfer_stages(),
            vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::HOST,
            vk::AccessFlags2::HOST_READ,
        );
        unsafe { self.context.device().end_command_buffer(self.command_buffer())? };
        Ok(())
    }

    fn submit_segment(&mut self) -> Result<(), Error> {
        self.end_segment()?;
        let next = self.context.submitter().acquire()?;
        let ended = self.resources.replace(next).expect("encoding owns resources until it ends");
        let submitted =
            unsafe { VkCommandBufferSubmission::submit(self.context.clone(), ended, take(&mut self.retained)) };
        self.reads.clear();
        self.writes.clear();
        self.segment_work = 0;
        self.begin_segment()?;
        self.segments.push(submitted?);
        Ok(())
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
            self.context.submitter().recycle(resources);
        }
        self.context.submitter().unsubmitted.fetch_sub(1, Ordering::Relaxed);
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
        let alignment = usize::min(size, 64).next_power_of_two();
        self.uploads.allocate(size, alignment, |size| VkBufferCpuGpu::new(context, size, VkBufferMemory::Staging))
    }

    fn allocate_scratch(
        &mut self,
        size: usize,
    ) -> Result<<Vulkan as Backend>::ScratchBuffer, Error> {
        let context = self.context.clone();
        self.allocation_pool.allocate(size, |size| VkBuffer::new(context, size))
    }

    fn encode_copy(
        &mut self,
        src: impl BufferRef<Backend = Vulkan>,
        dst: impl BufferMut<Backend = Vulkan>,
    ) {
        let (source, source_range) = src.parts();
        let (destination, destination_range) = dst.parts();
        assert_eq!(source_range.iter().len(), destination_range.iter().len(), "copy ranges differ in size");
        let (source, source_range) = VkBuffer::resolve(source, source_range);
        let (destination, destination_range) = VkBuffer::resolve(destination, destination_range);
        let (Some(from), Some(to)) = (
            device_addresses(source, &source_range).expect("Vulkan copy"),
            device_addresses(destination, &destination_range).expect("Vulkan copy"),
        ) else {
            return;
        };
        assert!(from.end <= to.start || to.end <= from.start, "copy ranges overlap within the same buffer");
        if !self.access([from].into_iter(), [to].into_iter()) {
            return;
        }
        let region = vk::BufferCopy::default()
            .src_offset(source_range.start)
            .dst_offset(destination_range.start)
            .size(source_range.end - source_range.start);
        unsafe {
            self.context.device().cmd_copy_buffer(
                self.command_buffer(),
                source.buffer(),
                destination.buffer(),
                &[region],
            );
        }
        self.retained.push(source.clone());
        self.retained.push(destination.clone());
    }

    fn encode_fill(
        &mut self,
        dst: impl BufferMut<Backend = Vulkan>,
        value: u8,
    ) {
        let (destination, range) = dst.parts();
        let (destination, range) = VkBuffer::resolve(destination, range);
        let Some(addresses) = device_addresses(destination, &range).expect("Vulkan fill") else {
            return;
        };
        assert!(
            range.start.is_multiple_of(4) && range.end.is_multiple_of(4),
            "fill range {range:?} is not 4-byte aligned"
        );
        if !self.access(std::iter::empty(), [addresses].into_iter()) {
            return;
        }
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
        if let Some(debug_utils) = self.debug_utils() {
            let name = CString::new(name).expect("debug label contains NUL");
            let label = vk::DebugUtilsLabelEXT::default().label_name(&name);
            unsafe { debug_utils.cmd_insert_debug_utils_label(self.command_buffer(), &label) };
        }
    }

    fn with_timestamps(self) -> Self {
        unimplemented!()
    }

    fn end_encoding(mut self) -> VkCommandBufferExecutable {
        if self.named {
            self.end_debug_label();
        }
        let failure = self.failure.take().or_else(|| self.end_segment().err().map(Error::from));
        let resources = self.resources.take().expect("encoding owns resources until it ends");
        unsafe {
            VkCommandBufferExecutable::new(
                self.context.clone(),
                resources,
                take(&mut self.retained),
                take(&mut self.segments),
                failure,
            )
        }
    }
}

fn shader_and_transfer_stages() -> vk::PipelineStageFlags2 {
    vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::ALL_TRANSFER
}

fn device_addresses(
    buffer: &VkBuffer,
    range: &Range<u64>,
) -> Result<Option<Range<u64>>, VkCommandBufferError> {
    if range.start > range.end || range.end > buffer.size() {
        return Err(VkCommandBufferError::BufferRange {
            start: range.start,
            end: range.end,
            size: buffer.size(),
        });
    }
    Ok((!range.is_empty()).then(|| buffer.device_address() + range.start..buffer.device_address() + range.end))
}

pub struct VkDebugGroupSpan<'a> {
    encoding: &'a mut VkCommandBufferEncoding,
}

impl<'a> VkDebugGroupSpan<'a> {
    pub fn new(
        encoding: &'a mut VkCommandBufferEncoding,
        name: &str,
    ) -> Self {
        encoding.begin_debug_label(name);
        Self {
            encoding,
        }
    }
}

impl Deref for VkDebugGroupSpan<'_> {
    type Target = VkCommandBufferEncoding;

    fn deref(&self) -> &Self::Target {
        self.encoding
    }
}

impl DerefMut for VkDebugGroupSpan<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.encoding
    }
}

impl Drop for VkDebugGroupSpan<'_> {
    fn drop(&mut self) {
        self.encoding.end_debug_label();
    }
}
