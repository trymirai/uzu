use std::{
    ffi::CString,
    mem::take,
    ops::Range,
    sync::{Arc, atomic::Ordering},
    time::Duration,
};

use ash::vk;
use bytemuck::NoUninit;
use futures::channel::oneshot;

use crate::backends::{
    common::{
        Backend, BufferMut, BufferRef, CommandBufferEncoding,
        allocator::{bump::BumpAllocator, pool::PoolAllocator},
    },
    vulkan::{
        Error, VkBuffer, VkBufferMemory, VkCommandBuffer, VkCommandBufferError, VkCommandBufferExecutable,
        VkCommandBufferResources, VkCommandBufferSubmission, VkComputePipeline, VkContext, VkDebugGroupSpan,
        VkDeviceCapabilities, VkTimestampRecorder, Vulkan,
    },
};

/// Smallest buffer that holds uploads, so small uploads of one command buffer share it.
const UPLOAD_PAGE_SIZE: usize = 4096;
/// Commands of the first segment; each later one holds twice as many as the one before, for up to this many doublings.
const FIRST_SEGMENT_COMMANDS: usize = 32;
const SEGMENT_DOUBLINGS: usize = 10;
/// Multiply-add FLOPs a segment may hold per compute unit. Even at one unit's scalar rate, a few hundred GFLOPS, a
/// segment then runs for tens of milliseconds, far inside the two-second watchdogs that reset a GPU whose work cannot
/// be preempted.
const SEGMENT_WORK_PER_UNIT: u64 = 8_000_000_000;
/// The units assumed where the device does not report them.
const UNREPORTED_UNITS: u64 = 8;

/// A recording primary command buffer that retains every buffer and pipeline it references and inserts barriers
/// where accesses conflict. It records in segments, each submitted once full while recording goes on, so the GPU
/// starts on a long recording instead of waiting for its end; segments growing twice as long keep the submissions
/// few, and a work budget the heavy kernels draw on keeps each one short of the GPU watchdog. Segments go early only
/// while no other command buffer of the context is unsubmitted, so none passes one recorded or ended before it; a
/// command buffer created after a segment went out runs after that segment, even if it is submitted before this one
/// ends, and dropping the encoding does not recall it. A device error ends the recording: later commands are skipped
/// and submission reports the error.
pub struct VkCommandBufferEncoding {
    context: Arc<VkContext>,
    resources: Option<VkCommandBufferResources>,
    retained: Vec<Arc<dyn Send + Sync>>,
    /// Device-address ranges read and written since the last barrier: a few at a time, as nearly every dispatch reads
    /// what the previous one wrote.
    reads: Vec<Range<u64>>,
    writes: Vec<Range<u64>>,
    /// Whether a barrier of the segment ordered the accesses before it for dispatches only, so a transfer has to
    /// wait for them.
    compute_only_barrier: bool,
    /// Uploads and constants: host-written memory of this command buffer, whose pages dispatches retain.
    uploads: BumpAllocator<Arc<VkBuffer>>,
    /// Scratch pages; a dispatch retains the pages it uses until the command buffer completes.
    allocation_pool: Arc<PoolAllocator<Arc<VkBuffer>>>,
    /// Completions of the segments submitted so far, and the commands and work recorded into the current one.
    segments: Vec<oneshot::Receiver<Result<(Duration, Box<[u64]>), Error>>>,
    segment_commands: usize,
    segment_work: u64,
    /// The work the next command starts, declared by its kernel, and the work a segment may hold.
    declared_work: u64,
    work_budget: u64,
    /// Debug-group spans, when recording timestamps.
    timestamps: Option<VkTimestampRecorder>,
    /// The device error that ended the recording.
    failure: Option<Error>,
}

impl VkCommandBufferEncoding {
    pub fn new(
        context: Arc<VkContext>,
        allocation_pool: Arc<PoolAllocator<Arc<VkBuffer>>>,
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
            uploads: BumpAllocator::new(UPLOAD_PAGE_SIZE),
            allocation_pool,
            segments: Vec::new(),
            segment_commands: 0,
            segment_work: 0,
            declared_work: 0,
            work_budget: units * SEGMENT_WORK_PER_UNIT,
            timestamps: None,
            failure: None,
        };
        encoding.begin_segment()?;
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
        let destination_end = destination_offset.checked_add(size).ok_or(VkCommandBufferError::BufferRange {
            start: destination_offset,
            end: u64::MAX,
            size: destination.as_ref().size(),
        })?;
        let destination_range = destination_offset..destination_end;
        let source_addresses = device_addresses(source, &source_range)?;
        let destination_addresses = device_addresses(destination, &destination_range)?;
        if source_addresses.start < destination_addresses.end && destination_addresses.start < source_addresses.end {
            return Err(VkCommandBufferError::CopyOverlap.into());
        }
        if !self.access(true, [source_addresses].into_iter(), [destination_addresses].into_iter()) {
            return Ok(());
        }
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
    pub fn upload<T: NoUninit>(
        &mut self,
        data: &[T],
    ) -> Result<Option<(Arc<VkBuffer>, Range<u64>)>, Error> {
        let size = size_of_val(data);
        if size == 0 {
            return Ok(None);
        }
        let context = &self.context;
        let allocation = self.uploads.allocate(size, |size| {
            Ok::<_, Error>(Arc::new(VkBuffer::new(context.clone(), size as u64, VkBufferMemory::Constants)?))
        })?;
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
                bytemuck::cast_slice::<T, u8>(data).as_ptr(),
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
            return Err(VkCommandBufferError::FillAlignment {
                start: range.start,
                end: range.end,
            }
            .into());
        }
        if !self.access(true, std::iter::empty(), [addresses].into_iter()) {
            return Ok(());
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
            return Err(VkCommandBufferError::DispatchGroups {
                groups,
                limit: limits.max_compute_work_group_count,
            }
            .into());
        }
        for (buffer, range) in buffers.clone() {
            if range.start > range.end || range.end > buffer.as_ref().size() {
                return Err(VkCommandBufferError::BufferRange {
                    start: range.start,
                    end: range.end,
                    size: buffer.as_ref().size(),
                }
                .into());
            }
        }
        let address = |(buffer, range): (&Arc<VkBuffer>, Range<u64>)| {
            (!range.is_empty()).then(|| buffer.device_address() + range.start..buffer.device_address() + range.end)
        };
        if !self.access(false, reads.filter_map(address), writes.filter_map(address)) {
            return Ok(());
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
        Ok(())
    }

    pub fn command_buffer(&self) -> vk::CommandBuffer {
        self.resources().command_buffer()
    }

    /// Whether commands still record: no device error has ended the recording.
    pub fn is_recording(&self) -> bool {
        self.failure.is_none()
    }

    fn resources(&self) -> &VkCommandBufferResources {
        self.resources.as_ref().expect("encoding owns resources until it ends")
    }

    /// Declares the multiply-add FLOPs of the commands a heavy kernel records next, so a segment closes before they
    /// would take it past its work budget.
    pub fn declare_work(
        &mut self,
        flops: u64,
    ) {
        self.declared_work = self.declared_work.saturating_add(flops);
    }

    /// Starts a span named `label` when recording timestamps.
    pub fn start_timestamp_span(
        &mut self,
        label: &str,
    ) {
        if let Some(timestamp) = self.write_span_timestamp() {
            self.timestamps.as_mut().expect("written while recording timestamps").start(label, timestamp);
        }
    }

    /// Ends the innermost span when recording timestamps.
    pub fn end_timestamp_span(&mut self) {
        if let Some(timestamp) = self.write_span_timestamp() {
            self.timestamps.as_mut().expect("written while recording timestamps").end(timestamp);
        }
    }

    /// Writes a span timestamp while recording timestamps and commands, and returns its segment and index there.
    fn write_span_timestamp(&mut self) -> Option<(usize, u32)> {
        if self.timestamps.is_none() || self.failure.is_some() {
            return None;
        }
        let resources = self.resources.as_mut().expect("encoding owns resources until it ends");
        match unsafe { resources.write_span_timestamp(self.context.device()) } {
            Ok(index) => Some((self.segments.len(), index)),
            Err(error) => {
                self.failure = Some(error);
                None
            },
        }
    }

    /// Prepares for one more command, a transfer or a dispatch: submits a segment that is full or that the command's
    /// declared work would take past its budget, which the next begins after, or otherwise orders the command after
    /// the accesses it conflicts with. False once a device error has ended the recording, when the command is skipped.
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
            || (self.segment_work > 0 && self.segment_work.saturating_add(work) > self.work_budget);
        if full && self.context.unsubmitted_command_buffers().load(Ordering::Relaxed) == 1 {
            if let Err(error) = self.submit_segment() {
                self.failure = Some(error);
                return false;
            }
        } else if (transfer && self.compute_only_barrier)
            || reads.clone().chain(writes.clone()).any(|range| overlaps(&self.writes, &range))
            || writes.clone().any(|range| overlaps(&self.reads, &range))
        {
            // A dispatch waits in the compute stage alone, which makes a dependent dispatch about a third cheaper on
            // NVIDIA than waiting in the transfer stage too; the rare transfers wait in both.
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
        self.segment_work = self.segment_work.saturating_add(work);
        true
    }

    /// Begins the current segment after all work submitted before it, with its start timestamp.
    fn begin_segment(&self) -> Result<(), Error> {
        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.context.device().begin_command_buffer(self.command_buffer(), &begin_info)? };
        self.memory_barrier(
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_WRITE,
            shader_and_transfer_stages(),
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        );
        unsafe { self.resources().write_timestamp(self.context.device(), 0) };
        // Execution-only dependency so recorded work cannot run before the start timestamp is written.
        self.memory_barrier(
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::NONE,
            shader_and_transfer_stages(),
            vk::AccessFlags2::NONE,
        );
        Ok(())
    }

    /// Ends the current segment with its writes visible to the host and its end timestamp.
    fn end_segment(&self) -> Result<(), Error> {
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

    /// Submits the current segment with what it retains and continues in a new one, which tracks hazards afresh.
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
            // Never submitted, so the GPU cannot be using it; the pool reset on reuse discards the recording.
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
        self.uploads
            .allocate(size, |size| Ok(Arc::new(VkBuffer::new(context, size as u64, VkBufferMemory::Constants)?)))
    }

    fn allocate_scratch(
        &mut self,
        size: usize,
    ) -> Result<<Vulkan as Backend>::ScratchBuffer, Error> {
        let context = self.context.clone();
        self.allocation_pool
            .allocate(size, |size| Ok(Arc::new(VkBuffer::new(context, size as u64, VkBufferMemory::Device)?)))
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
        if let Some(debug_utils) = self.context.debug_utils()
            && self.is_recording()
        {
            let name = CString::new(name).expect("debug point name contains NUL");
            let label = vk::DebugUtilsLabelEXT::default().label_name(&name);
            unsafe { debug_utils.cmd_insert_debug_utils_label(self.command_buffer(), &label) };
        }
    }

    /// Timing debug groups needs the device clock, so without calibrated timestamps the command buffer fails.
    fn with_timestamps(mut self) -> Self {
        if !self.context.physical_device().capabilities.contains(VkDeviceCapabilities::CALIBRATED_TIMESTAMPS) {
            self.failure.get_or_insert(VkCommandBufferError::UncalibratedTimestamps.into());
        }
        self.timestamps = Some(VkTimestampRecorder::default());
        self
    }

    fn end_encoding(mut self) -> VkCommandBufferExecutable {
        let failure = self.failure.take().or_else(|| self.end_segment().err());
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
) -> Result<Range<u64>, Error> {
    if range.start >= range.end || range.end > buffer.size() {
        return Err(VkCommandBufferError::BufferRange {
            start: range.start,
            end: range.end,
            size: buffer.size(),
        }
        .into());
    }
    Ok(buffer.device_address() + range.start..buffer.device_address() + range.end)
}
