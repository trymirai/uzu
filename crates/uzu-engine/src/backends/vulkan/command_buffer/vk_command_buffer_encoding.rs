use std::{
    ffi::CString,
    mem::take,
    ops::{Deref, DerefMut, Range},
    sync::Arc,
};

use ash::vk;
use bytemuck::NoUninit;
use futures::channel::oneshot;
use rangemap::RangeSet;

use crate::backends::{
    common::{
        Backend, Buffer, BufferMut, BufferRef, CommandBufferEncoding,
        allocator::{bump::BumpAllocator, pool::PoolAllocator},
    },
    vulkan::{
        Error, VkBuffer, VkBufferCpuGpu, VkCommandBuffer, VkCommandBufferExecutable, VkCommandPool, VkComputePipeline,
        VkContext, Vulkan, failed_segment,
    },
};

const UPLOAD_PAGE_SIZE: usize = 4096;
// The most declared FLOPs in one submission, for each core the GPU reports. An operating system resets a GPU that
// works on one submission for too long (Windows after 2 seconds), and this keeps a core as slow as 0.25 GFLOP/s
// within that: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/timeout-detection-and-recovery
const SEGMENT_WORK_PER_CORE: u64 = 500_000_000;
// The cores assumed for a GPU that reports no count.
const UNREPORTED_CORES: u32 = 16;

pub struct VkCommandBufferEncoding {
    context: Arc<VkContext>,
    command_pool: Option<VkCommandPool>,
    retained: Vec<Arc<dyn Send + Sync>>,
    reads: RangeSet<u64>,
    writes: RangeSet<u64>,
    uploads: BumpAllocator<VkBufferCpuGpu>,
    allocation_pool: Arc<PoolAllocator<Arc<VkBuffer>>>,
    segments: Vec<oneshot::Receiver<Result<(), Error>>>,
    segment_work: u64,
    named: bool,
}

impl VkCommandBufferEncoding {
    pub fn new(
        context: Arc<VkContext>,
        allocation_pool: Arc<PoolAllocator<Arc<VkBuffer>>>,
        name: Option<&str>,
    ) -> Result<Self, Error> {
        let command_pool = context.command_queue().acquire()?;
        let encoding = Self {
            command_pool: Some(command_pool),
            context,
            retained: Vec::new(),
            reads: RangeSet::new(),
            writes: RangeSet::new(),
            uploads: BumpAllocator::new(UPLOAD_PAGE_SIZE, 1),
            allocation_pool,
            segments: Vec::new(),
            segment_work: 0,
            named: name.is_some(),
        };
        encoding.begin_segment();
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
        if size == 0 {
            return None;
        }
        let allocation =
            self.allocate_constant_from_slice(bytemuck::cast_slice::<T, u8>(data)).expect("Vulkan upload allocation");
        let buffer = allocation.page().buffer().clone();
        let range = allocation.range();
        assert!(
            (buffer.device_address() + range.start as u64).is_multiple_of(align_of::<T>() as u64),
            "upload of {size} bytes is not aligned to {}",
            align_of::<T>()
        );
        Some((buffer, range.start as u64..range.end as u64))
    }

    pub fn encode_dispatch<'b>(
        &mut self,
        kernel: &'static str,
        pipeline: &Arc<VkComputePipeline>,
        push_constants: &[u8],
        groups: [u32; 3],
        reads: impl IntoIterator<Item = (&'b Arc<VkBuffer>, Range<u64>), IntoIter: Clone>,
        writes: impl IntoIterator<Item = (&'b Arc<VkBuffer>, Range<u64>), IntoIter: Clone>,
    ) {
        self.split_before(0);
        let reads = reads.into_iter();
        let writes = writes.into_iter();
        let limit = self.context.device().physical_device.limits.max_work_group_count;
        assert!(
            groups.iter().zip(limit).all(|(&count, limit)| count <= limit),
            "{kernel}: dispatch of {groups:?} groups exceeds the device limit {limit:?}"
        );
        let addresses = |(buffer, range): (&Arc<VkBuffer>, Range<u64>)| device_addresses(buffer, &range);
        self.access(reads.clone().filter_map(addresses), writes.clone().filter_map(addresses));
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
        self.retained.extend(reads.chain(writes).map(|(buffer, _)| buffer.clone() as Arc<dyn Send + Sync>));
    }

    fn command_buffer(&self) -> vk::CommandBuffer {
        self.command_pool.as_ref().expect("encoding owns its command pool until it ends").command_buffer()
    }

    fn begin_debug_label(
        &self,
        name: &str,
    ) {
        if let Some(debug_utils) = &self.context.device().debug_labels {
            let name = CString::new(name).expect("debug label contains NUL");
            let label = vk::DebugUtilsLabelEXT::default().label_name(&name);
            unsafe { debug_utils.cmd_begin_debug_utils_label(self.command_buffer(), &label) };
        }
    }

    fn end_debug_label(&self) {
        if let Some(debug_utils) = &self.context.device().debug_labels {
            unsafe { debug_utils.cmd_end_debug_utils_label(self.command_buffer()) };
        }
    }

    // A kernel declares its work before its dispatches. Work that would take the segment over the budget starts a
    // new one.
    pub fn declare_work(
        &mut self,
        flops: u64,
    ) {
        self.split_before(flops);
        self.segment_work += flops;
    }

    // Submits what is encoded so far when the GPU has nothing left to run, so it works while the rest is encoded, or
    // when `flops` more would exceed the cap. Splitting before the work is counted keeps it with its own dispatch.
    fn split_before(
        &mut self,
        flops: u64,
    ) {
        let cores = self.context.device().physical_device.gpu_core_count.unwrap_or(UNREPORTED_CORES);
        if !self.retained.is_empty()
            && (self.context.command_queue().is_idle()
                || self.segment_work + flops > u64::from(cores) * SEGMENT_WORK_PER_CORE)
        {
            self.submit_segment();
        }
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

    fn begin_segment(&self) {
        self.memory_barrier(
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_WRITE,
            shader_and_transfer_stages(),
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        );
    }

    fn end_segment(&self) {
        self.memory_barrier(
            shader_and_transfer_stages(),
            vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::HOST,
            vk::AccessFlags2::HOST_READ,
        );
    }

    // Submits the current command buffer and begins the next, unless another command buffer is still unsubmitted and
    // has to reach the queue first. Submitting first lets the GPU start without waiting for the next command buffer
    // to be reset and begun.
    fn submit_segment(&mut self) {
        if !self.context.command_queue().has_single_unsubmitted() {
            return;
        }
        self.end_segment();
        let ended = self.command_pool.take().expect("encoding owns its command pool until it ends");
        let submitted = self.context.command_queue().submit(self.context.clone(), ended, take(&mut self.retained));
        // Like Metal's encoder creation, a command buffer that cannot be begun mid-encoding is fatal.
        self.command_pool =
            Some(self.context.command_queue().acquire().expect("Vulkan command buffer for the next segment"));
        self.reads.clear();
        self.writes.clear();
        self.segment_work = 0;
        self.begin_segment();
        self.segments.push(submitted.unwrap_or_else(|error| failed_segment(error.into())));
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
        if let Some(command_pool) = self.command_pool.take() {
            self.context.command_queue().recycle(command_pool);
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
        let alignment = usize::min(size, 64).next_power_of_two();
        self.uploads.allocate(size, alignment, |size| VkBufferCpuGpu::staging(self.context.clone(), size))
    }

    fn allocate_scratch(
        &mut self,
        size: usize,
    ) -> Result<<Vulkan as Backend>::ScratchBuffer, Error> {
        self.allocation_pool.allocate(size, |size| VkBuffer::new(&self.context, size))
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
        let (Some(from), Some(to)) =
            (device_addresses(source, &source_range), device_addresses(destination, &destination_range))
        else {
            return;
        };
        assert!(from.end <= to.start || to.end <= from.start, "copy ranges overlap within the same buffer");
        self.access([from].into_iter(), [to].into_iter());
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
        let Some(addresses) = device_addresses(destination, &range) else {
            return;
        };
        assert!(
            range.start.is_multiple_of(4) && range.end.is_multiple_of(4),
            "fill range {range:?} is not 4-byte aligned"
        );
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
    }

    fn debug_group(
        &mut self,
        name: &str,
    ) -> VkDebugGroupSpan<'_> {
        self.begin_debug_label(name);
        VkDebugGroupSpan {
            encoding: self,
        }
    }

    fn debug_point(
        &mut self,
        name: &str,
    ) {
        if let Some(debug_utils) = &self.context.device().debug_labels {
            let name = CString::new(name).expect("debug label contains NUL");
            let label = vk::DebugUtilsLabelEXT::default().label_name(&name);
            unsafe { debug_utils.cmd_insert_debug_utils_label(self.command_buffer(), &label) };
        }
    }

    fn with_timestamps(self) -> Self {
        unimplemented!()
    }

    fn end_encoding(mut self) -> VkCommandBufferExecutable {
        assert!(self.uploads.is_done(), "attempted to end encoding while constants are still alive");
        if self.named {
            self.end_debug_label();
        }
        self.end_segment();
        let command_pool = self.command_pool.take().expect("encoding owns its command pool until it ends");
        VkCommandBufferExecutable::new(
            self.context.clone(),
            command_pool,
            take(&mut self.retained),
            take(&mut self.segments),
        )
    }
}

fn shader_and_transfer_stages() -> vk::PipelineStageFlags2 {
    vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::ALL_TRANSFER
}

fn device_addresses(
    buffer: &Arc<VkBuffer>,
    range: &Range<u64>,
) -> Option<Range<u64>> {
    assert!(
        range.start <= range.end && range.end <= buffer.size() as u64,
        "buffer range {range:?} is reversed or exceeds a buffer of {} bytes",
        buffer.size()
    );
    (!range.is_empty()).then(|| buffer.device_address() + range.start..buffer.device_address() + range.end)
}

pub struct VkDebugGroupSpan<'a> {
    encoding: &'a mut VkCommandBufferEncoding,
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
