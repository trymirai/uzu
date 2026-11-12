use std::{
    ffi::CString,
    mem::take,
    ops::Range,
    sync::{Arc, atomic::Ordering},
};

use ash::{ext, vk};
use bytemuck::NoUninit;
use futures::channel::oneshot;

use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding,
            allocator::{bump::BumpAllocator, pool::PoolAllocator},
        },
        vulkan::{
            BlitCopyVulkanKernel, Error, VkBuffer, VkBufferMemory, VkCommandBuffer, VkCommandBufferError,
            VkCommandBufferExecutable, VkCommandBufferResources, VkCommandBufferSubmission, VkComputePipeline,
            VkContext, VkDebugGroupSpan, VkSegmentResult, VkTimestampRecorder, Vulkan,
        },
    },
    data_type::DataType,
};

const UPLOAD_PAGE_SIZE: usize = 4096;
const FIRST_SEGMENT_COMMANDS: usize = 32;
const SEGMENT_DOUBLINGS: usize = 10;
const SEGMENT_WORK_PER_UNIT: u64 = 8_000_000_000;
const UNREPORTED_UNITS: u64 = 8;

pub struct VkCommandBufferEncoding {
    context: Arc<VkContext>,
    resources: Option<VkCommandBufferResources>,
    retained: Vec<Arc<dyn Send + Sync>>,
    reads: Vec<Range<u64>>,
    writes: Vec<Range<u64>>,
    compute_only_barrier: bool,
    uploads: BumpAllocator<Arc<VkBuffer>>,
    allocation_pool: Arc<PoolAllocator<Arc<VkBuffer>>>,
    blit_copy: [Option<BlitCopyVulkanKernel>; 2],
    segments: Vec<oneshot::Receiver<VkSegmentResult>>,
    segment_commands: usize,
    segment_work: u64,
    declared_work: u64,
    work_budget: u64,
    timestamps: Option<VkTimestampRecorder>,
    named: bool,
    failure: Option<Error>,
}

impl VkCommandBufferEncoding {
    pub fn new(
        context: Arc<VkContext>,
        allocation_pool: Arc<PoolAllocator<Arc<VkBuffer>>>,
        name: Option<&str>,
    ) -> Result<Self, Error> {
        let resources = context.acquire_command_buffer_resources()?;
        context.unsubmitted_command_buffers().fetch_add(1, Ordering::Relaxed);
        let units = context.physical_device().compute_units.map_or(UNREPORTED_UNITS, u64::from);
        let encoding = Self {
            resources: Some(resources),
            context,
            retained: Vec::new(),
            reads: Vec::new(),
            writes: Vec::new(),
            compute_only_barrier: false,
            uploads: BumpAllocator::new(UPLOAD_PAGE_SIZE, 1),
            allocation_pool,
            blit_copy: [None, None],
            segments: Vec::new(),
            segment_commands: 0,
            segment_work: 0,
            declared_work: 0,
            work_budget: units * SEGMENT_WORK_PER_UNIT,
            timestamps: None,
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
        let buffer = allocation.page().clone();
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
        let limits = &self.context.physical_device().properties.limits;
        let checked = if groups.iter().zip(limits.max_compute_work_group_count).any(|(&count, limit)| count > limit) {
            Err(VkCommandBufferError::DispatchGroups {
                groups,
                limit: limits.max_compute_work_group_count,
            })
        } else {
            buffers.clone().try_for_each(|(buffer, range)| device_addresses(buffer, &range).map(drop))
        };
        if let Err(error) = checked {
            self.failure.get_or_insert(VkCommandBufferError::Dispatch(kernel, Box::new(error)).into());
            return;
        }
        let address = |(buffer, range): (&Arc<VkBuffer>, Range<u64>)| device_addresses(buffer, &range).ok().flatten();
        if !self.access(false, reads.filter_map(address), writes.filter_map(address)) {
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
        self.resources().command_buffer()
    }

    fn resources(&self) -> &VkCommandBufferResources {
        self.resources.as_ref().expect("encoding owns resources until it ends")
    }

    fn debug_utils(&self) -> Option<&ext::debug_utils::Device> {
        self.context.debug_utils().filter(|_| self.failure.is_none())
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

    pub fn start_timestamp_span(
        &mut self,
        label: &str,
    ) {
        if let Some(timestamp) = self.write_span_timestamp() {
            self.timestamps.as_mut().expect("written while recording timestamps").start(label, timestamp);
        }
    }

    pub fn end_timestamp_span(&mut self) {
        if let Some(timestamp) = self.write_span_timestamp() {
            self.timestamps.as_mut().expect("written while recording timestamps").end(timestamp);
        }
    }

    fn write_span_timestamp(&mut self) -> Option<(usize, u32)> {
        if self.timestamps.is_none() || self.failure.is_some() {
            return None;
        }
        let resources = self.resources.as_mut().expect("encoding owns resources until it ends");
        match unsafe { resources.write_span_timestamp(self.context.device()) } {
            Ok(index) => Some((self.segments.len(), index)),
            Err(error) => {
                self.failure = Some(error.into());
                None
            },
        }
    }

    fn access(
        &mut self,
        transfer: bool,
        reads: impl Iterator<Item = Range<u64>> + Clone,
        writes: impl Iterator<Item = Range<u64>> + Clone,
    ) -> bool {
        if self.failure.is_some() {
            return false;
        }
        let overlaps = |tracked: &[Range<u64>], range: &Range<u64>| {
            tracked.iter().any(|other| other.start < range.end && range.start < other.end)
        };
        let work = take(&mut self.declared_work);
        let full = self.segment_commands >= FIRST_SEGMENT_COMMANDS << self.segments.len().min(SEGMENT_DOUBLINGS)
            || (self.segment_work > 0 && self.segment_work + work > self.work_budget);
        if full && self.context.unsubmitted_command_buffers().load(Ordering::Relaxed) == 1 {
            if let Err(error) = self.submit_segment() {
                self.failure = Some(error);
                return false;
            }
        } else if (transfer && self.compute_only_barrier)
            || reads.clone().chain(writes.clone()).any(|range| overlaps(&self.writes, &range))
            || writes.clone().any(|range| overlaps(&self.reads, &range))
        {
            let (stages, accesses) = if transfer {
                (
                    shader_and_transfer_stages(),
                    vk::AccessFlags2::SHADER_STORAGE_READ
                        | vk::AccessFlags2::SHADER_STORAGE_WRITE
                        | vk::AccessFlags2::TRANSFER_READ
                        | vk::AccessFlags2::TRANSFER_WRITE,
                )
            } else {
                (
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                )
            };
            self.memory_barrier(
                shader_and_transfer_stages(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::TRANSFER_WRITE,
                stages,
                accesses,
            );
            self.compute_only_barrier = !transfer;
            self.reads.clear();
            self.writes.clear();
        }
        self.reads.extend(reads);
        self.writes.extend(writes);
        self.segment_commands += 1;
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
        unsafe { self.resources().write_timestamp(self.context.device(), 0) };
        self.memory_barrier(
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::NONE,
            shader_and_transfer_stages(),
            vk::AccessFlags2::NONE,
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
        let device = self.context.device();
        unsafe {
            self.resources().write_timestamp(device, 1);
            device.end_command_buffer(self.command_buffer())?;
        }
        Ok(())
    }

    fn submit_segment(&mut self) -> Result<(), Error> {
        self.end_segment()?;
        let next = self.context.acquire_command_buffer_resources()?;
        let ended = self.resources.replace(next).expect("encoding owns resources until it ends");
        let submitted =
            unsafe { VkCommandBufferSubmission::submit(self.context.clone(), ended, take(&mut self.retained)) };
        self.reads.clear();
        self.writes.clear();
        self.compute_only_barrier = false;
        self.segment_commands = 0;
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
            unsafe { self.context.recycle_command_buffer_resources(resources) };
        }
        self.context.unsubmitted_command_buffers().fetch_sub(1, Ordering::Relaxed);
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
        self.uploads.allocate(size, alignment, |size| VkBuffer::new(context, size, VkBufferMemory::Constants))
    }

    fn allocate_scratch(
        &mut self,
        size: usize,
    ) -> Result<<Vulkan as Backend>::ScratchBuffer, Error> {
        let context = self.context.clone();
        self.allocation_pool.allocate(size, |size| VkBuffer::new(context, size, VkBufferMemory::Device))
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
        if self.failure.is_some() {
            return;
        }
        let size = from.end - from.start;
        let words = usize::from([from.start, to.start, size].iter().all(|value| value.is_multiple_of(4)));
        let kernel = match self.blit_copy[words].take() {
            Some(kernel) => kernel,
            None => match BlitCopyVulkanKernel::new(&self.context, [DataType::U8, DataType::U32][words]) {
                Ok(kernel) => kernel,
                Err(error) => {
                    self.failure = Some(error);
                    return;
                },
            },
        };
        let length = u32::try_from(size >> (2 * words)).expect("copy of more than u32::MAX elements");
        unsafe { kernel.encode((source, source_range), (destination, destination_range), length, self) };
        self.blit_copy[words] = Some(kernel);
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
        if !self.access(true, std::iter::empty(), [addresses].into_iter()) {
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

    fn with_timestamps(mut self) -> Self {
        self.timestamps = Some(VkTimestampRecorder::default());
        self
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
                self.timestamps.take(),
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
