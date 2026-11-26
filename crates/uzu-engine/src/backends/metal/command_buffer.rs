use std::{
    ops::{Deref, DerefMut},
    range::Range,
    sync::Arc,
    time::Duration,
};

use futures::channel::mpsc;
use metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer, MTL4CommandBufferExt,
    MTL4CommandEncoder, MTL4CommandEncoderExt, MTL4CommandQueue, MTL4CommandQueueExt, MTL4CommitFeedback,
    MTL4CommitFeedbackExt, MTL4CommitFeedbackHandler, MTL4CommitOptions, MTL4ComputeCommandEncoder,
    MTL4VisibilityOptions, MTLDeviceExt, MTLSharedEvent, MTLStages,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use parking_lot::{Mutex, MutexGuard};
use rangemap::RangeSet;
use smallvec::SmallVec;

use crate::backends::{
    common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding,
        CommandBufferExecutable, CommandBufferPending, Context, TimestampSpan, allocator::bump::BumpAllocator,
    },
    metal::{
        Metal,
        buffer::MetalBufferExt,
        context::MetalContext,
        error::MetalError,
        kernel::{BlitCopyMetalKernel, BlitFillMetalKernel},
        timestamp_recorder::MetalTimestampRecorder,
    },
};

pub struct MetalCommandBuffer;

impl CommandBuffer for MetalCommandBuffer {
    type Backend = Metal;

    type Encoding = MetalCommandBufferEncoding;
    type Executable = MetalCommandBufferExecutable;
    type Pending = MetalCommandBufferPending;
    type Completed = MetalCommandBufferCompleted;
}

pub(super) struct Access {
    pub(super) range: Range<u64>,
    pub(super) write: bool,
}

/// An encoding is cut into segments (command buffers committed one after another) by the events it
/// signals and waits on. `Segments` gathers their reports and finishes the encoding once all are in.
/// [`Self::fail`], callable from any thread, makes awaiting the encoding fail instead of returning
/// wrong results: call it before the failed work is released to the queue, so the encoding's last
/// segment sees it.
#[derive(Default)]
pub struct Segments {
    state: Mutex<SegmentsState>,
}

struct SegmentsState {
    error: Option<String>,
    ane_scope: ane::w8a8::QueueScope,
    /// First GPU start and last GPU end of the segments reported so far.
    gpu_span: (f64, f64),
    /// Committed segments that have not reported yet.
    unreported: usize,
    /// Finishes the encoding once every committed segment has reported.
    finish: Option<Box<dyn FnOnce(Option<String>, Duration)>>,
}

impl Default for SegmentsState {
    fn default() -> Self {
        Self {
            error: None,
            ane_scope: ane::w8a8::QueueScope::default(),
            gpu_span: (f64::MAX, f64::MIN),
            unreported: 0,
            finish: None,
        }
    }
}

// SAFETY: every access to the report state is protected by `state`. `run_finish` takes the
// finalizer while holding that mutex, so it can run at most once, then releases the mutex before
// invoking the finalizer and dropping its captures. `finish_when_reported` accepts non-`Send`
// values, so this implementation relies on its module-private callers: the encoding and
// executable drop paths retain only GPU allocators and their owned allocation pool with no-op
// finalizers; submit retains those resources plus the Metal context, completed timestamp recorder,
// and channel sender. Allocator and pool recycling goes through mutex-protected caches, and their
// owned GPU allocations can be released on the callback thread. The submit finalizer resolves
// timestamps only after completion through the Metal device, then sends through the thread-safe
// channel. The feedback handler returns a retained allocator handle to the context's
// mutex-protected cache and keeps the committed command buffer alive through the callback. These
// concrete captures are safe to use or drop on the callback thread; arbitrary `R` or finalizers
// are not.
unsafe impl Send for Segments {}
unsafe impl Sync for Segments {}

impl Segments {
    pub(super) fn ane_scope(&self) -> ane::w8a8::QueueScope {
        self.state.lock().ane_scope.clone()
    }

    pub fn fail(
        &self,
        message: &str,
    ) {
        self.state.lock().error.get_or_insert_with(|| message.to_string());
    }

    fn report(
        &self,
        error: Option<String>,
        start: f64,
        end: f64,
    ) {
        let mut state = self.state.lock();
        if let Some(error) = error {
            state.error.get_or_insert(error);
        }
        state.gpu_span = (state.gpu_span.0.min(start), state.gpu_span.1.max(end));
        state.unreported -= 1;
        Self::run_finish(state);
    }

    /// Commits a finished segment and returns its allocator only after Metal reports completion.
    fn commit(
        self: &Arc<Self>,
        context: &Arc<MetalContext>,
        segment: Segment,
    ) {
        self.state.lock().unreported += 1;

        let command_buffer = segment.command_buffer.clone();
        let command_allocator = segment.allocator.clone();
        let context_alive = context.clone();
        let segments = self.clone();
        let feedback_handler = move |feedback: &ProtocolObject<dyn MTL4CommitFeedback>| {
            let error = feedback.error().map(|error| error.to_string());
            segments.report(error, feedback.gpu_start_time(), feedback.gpu_end_time());
            context_alive.command_allocator_cache.lock().push(command_allocator.clone());
            let _keep_alive = &segment.command_buffer;
        };
        let options = MTL4CommitOptions::new();
        options.add_feedback_handler(&MTL4CommitFeedbackHandler::new(feedback_handler));
        sync_sparse_queue(context);
        context.command_queue.commit_with_options(&[&command_buffer], &options);
    }

    /// Retains resources until all committed segments have reported, then releases them once.
    fn finish_when_reported<R: 'static>(
        &self,
        resources: R,
        finish: impl FnOnce(Option<String>, Duration) + 'static,
    ) {
        let mut state = self.state.lock();
        debug_assert!(state.finish.is_none());
        state.finish = Some(Box::new(move |error, gpu_time| {
            finish(error, gpu_time);
            drop(resources);
        }));
        Self::run_finish(state);
    }

    fn run_finish(mut state: MutexGuard<SegmentsState>) {
        if state.unreported > 0 {
            return;
        }
        let Some(finish) = state.finish.take() else {
            return;
        };
        let scope = state.ane_scope.clone();
        let error = state.error.take().or_else(|| scope.error().map(|error| {
            format!("ANE request failed during prefill: {error} (the Neural Engine is retired; reset the model state or restart the prefill, it runs on the GPU)")
        }));
        let (start, end) = state.gpu_span;
        drop(state);
        // SAFETY: every committed segment has reported and the encoding is closed.
        unsafe { scope.complete() };
        // Nothing reported: the span is empty and the time zero.
        finish(error, Duration::from_secs_f64((end - start).max(0.0)));
    }
}

/// The active command-buffer allocation and command buffer always move together.
struct Segment {
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
}

type SegmentResources = (BumpAllocator<<Metal as Backend>::GlobalGpuBuffer>, Arc<<Metal as Backend>::AllocationPool>);

/// Ordinary transformer paths reach at most four nested labels, and the longest non-ANE label is
/// "per layer embedding projection" (30 UTF-8 bytes). Six entries and 32 bytes cover those paths
/// with headroom; deeper or longer labels spill.
const DEBUG_GROUP_INLINE_DEPTH: usize = 6;
const DEBUG_GROUP_INLINE_BYTES: usize = 32;

#[derive(Default)]
struct DebugGroups(SmallVec<[SmallVec<[u8; DEBUG_GROUP_INLINE_BYTES]>; DEBUG_GROUP_INLINE_DEPTH]>);

impl DebugGroups {
    fn push(
        &mut self,
        name: &str,
    ) {
        let mut label = SmallVec::new();
        label.extend(name.as_bytes().iter().copied());
        self.0.push(label);
    }

    fn pop(&mut self) {
        let _ = self.0.pop();
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn iter(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|label| std::str::from_utf8(label).expect("debug-group labels are UTF-8"))
    }

    #[cfg(test)]
    fn is_inline(&self) -> bool {
        !self.0.spilled() && self.0.iter().all(|label| !label.spilled())
    }
}

pub struct MetalCommandBufferEncoding {
    active: Option<Segment>,
    pub(super) compute_encoder: Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>,
    pub(super) argument_table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    reads: RangeSet<u64>,
    writes: RangeSet<u64>,
    constant_allocator: Option<BumpAllocator<<Metal as Backend>::GlobalGpuBuffer>>,
    allocation_pool: Arc<<Metal as Backend>::AllocationPool>,
    pub(super) context: Arc<MetalContext>,
    timestamps: Option<MetalTimestampRecorder>,
    label: Option<String>,
    /// The open debug groups, to open them again in the next segment.
    debug_groups: DebugGroups,
    segments: Arc<Segments>,
}

unsafe impl Send for MetalCommandBufferEncoding {}

impl MetalCommandBufferEncoding {
    pub fn new(
        context: Arc<MetalContext>,
        name: Option<&str>,
        allocation_pool: Option<Arc<<Metal as Backend>::AllocationPool>>,
    ) -> Result<Self, MetalError> {
        let argument_table_descriptor = MTL4ArgumentTableDescriptor::new();
        argument_table_descriptor.set_label(name.map(|name| format!("{name} argument table")).as_deref());
        argument_table_descriptor.set_max_buffer_bind_count(31);
        let argument_table = context
            .device
            .new_argument_table_with_descriptor(&argument_table_descriptor)
            .map_err(|error| MetalError::CannotCreateArgumentTable(error.to_string()))?;
        let (active, compute_encoder) = begin_segment(&context, &argument_table, name)?;

        let constant_allocator = BumpAllocator::new(256 * 1024, 1);

        let allocation_pool = allocation_pool.unwrap_or_else(|| context.create_allocation_pool());

        Ok(Self {
            active: Some(active),
            compute_encoder,
            argument_table,
            reads: RangeSet::new(),
            writes: RangeSet::new(),
            constant_allocator: Some(constant_allocator),
            allocation_pool,
            timestamps: None,
            label: name.map(str::to_owned),
            debug_groups: DebugGroups::default(),
            segments: Arc::default(),
            context,
        })
    }

    /// Commits what is encoded so far, then makes the queue signal `event` with `value`.
    pub fn signal_event(
        &mut self,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        value: u64,
    ) -> Result<(), MetalError> {
        self.commit_segment(MTL4VisibilityOptions::Device | MTL4VisibilityOptions::ResourceAlias)?;
        self.context.command_queue.signal_event_value(ProtocolObject::from_ref(event), value);
        self.next_segment()
    }

    /// Commits what is encoded so far, then makes the queue wait for `event` to reach `value`:
    /// everything encoded after this runs after that.
    pub fn wait_event(
        &mut self,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        value: u64,
    ) -> Result<(), MetalError> {
        self.commit_segment(MTL4VisibilityOptions::Device)?;
        self.context.command_queue.wait_for_event_value(ProtocolObject::from_ref(event), value);
        self.next_segment()
    }

    pub fn failure(&self) -> Arc<Segments> {
        self.segments.clone()
    }

    /// Ends the current segment: the commit. Fails if there is none, after a segment could not be
    /// started.
    fn commit_segment(
        &mut self,
        visibility_options: MTL4VisibilityOptions,
    ) -> Result<(), MetalError> {
        let Some(segment) = self.active.take() else {
            return Err(MetalError::CommandBufferExecution("no open segment to commit".to_string()));
        };
        for _ in 0..self.debug_groups.len() {
            self.compute_encoder.pop_debug_group();
        }
        self.finish_command_buffer(&segment, visibility_options);
        self.segments.commit(&self.context, segment);
        Ok(())
    }

    /// Starts the next segment, reopening the debug groups.
    fn next_segment(&mut self) -> Result<(), MetalError> {
        let (segment, compute_encoder) = begin_segment(&self.context, &self.argument_table, self.label.as_deref())
            .inspect_err(|error| {
                // The committed segment stays closed: nothing is encoded or committed after it.
                self.segments.fail(&error.to_string());
            })?;
        self.active = Some(segment);
        self.compute_encoder = compute_encoder;
        self.reads.clear();
        self.writes.clear();
        for name in self.debug_groups.iter() {
            ProtocolObject::<dyn MTL4CommandEncoder>::push_debug_group(self.compute_encoder.as_ref(), name);
        }
        Ok(())
    }

    fn finish_command_buffer(
        &self,
        segment: &Segment,
        visibility_options: MTL4VisibilityOptions,
    ) {
        self.compute_encoder.barrier_after_stages_before_queue_stages_visibility_options(
            MTLStages::Dispatch,
            MTLStages::Dispatch | MTLStages::ResourceState,
            visibility_options,
        );
        self.compute_encoder.end_encoding();
        segment.command_buffer.end_command_buffer();
    }

    pub(super) fn access(
        &mut self,
        accesses: &[Access],
    ) {
        // TODO: more fine grained barriers
        if accesses.iter().any(|access| {
            self.writes.overlaps(&access.range.into()) || (access.write && self.reads.overlaps(&access.range.into()))
        }) {
            self.compute_encoder.barrier_after_encoder_stages_before_encoder_stages_visibility_options(
                MTLStages::Dispatch,
                MTLStages::Dispatch,
                MTL4VisibilityOptions::Device,
            );
            self.reads.clear();
            self.writes.clear();
        }

        for access in accesses {
            if access.write {
                self.writes.insert(access.range.into());
            } else {
                self.reads.insert(access.range.into());
            }
        }
    }
}

/// A new command buffer with its allocator and a compute encoder that waits for the queue's earlier work.
fn begin_segment(
    context: &MetalContext,
    argument_table: &ProtocolObject<dyn MTL4ArgumentTable>,
    name: Option<&str>,
) -> Result<(Segment, Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>), MetalError> {
    let command_allocator = if let Some(command_allocator) = context.command_allocator_cache.lock().pop() {
        command_allocator.reset();
        command_allocator
    } else {
        context.device.new_command_allocator().ok_or(MetalError::CannotCreateCommandBuffer)?
    };

    let Some(command_buffer) = context.device.new_mtl4_command_buffer() else {
        context.command_allocator_cache.lock().push(command_allocator);
        return Err(MetalError::CannotCreateCommandBuffer);
    };
    command_buffer.begin_command_buffer_with_allocator(&command_allocator);
    command_buffer.set_label(name);

    let compute_encoder = command_buffer.compute_command_encoder().unwrap();
    ProtocolObject::<dyn MTL4CommandEncoder>::set_label(compute_encoder.as_ref(), name);
    compute_encoder.barrier_after_queue_stages_before_stages_visibility_options(
        MTLStages::Dispatch | MTLStages::ResourceState,
        MTLStages::Dispatch,
        MTL4VisibilityOptions::Device,
    );
    compute_encoder.set_argument_table(Some(argument_table));
    Ok((
        Segment {
            allocator: command_allocator,
            command_buffer,
        },
        compute_encoder,
    ))
}

/// Makes the queue wait for the sparse mappings made since the last commit.
fn sync_sparse_queue(context: &MetalContext) {
    let mut sparse_state_locked = context.sparse_state.lock();
    if sparse_state_locked.did_ops {
        sparse_state_locked.did_ops = false;
        sparse_state_locked.event_value += 1;
        let value = sparse_state_locked.event_value;
        context.sparse_queue.signal_event_value(&context.sparse_event, value);
        context.command_queue.wait_for_event_value(&context.sparse_event, value);
    }
}

impl CommandBufferEncoding for MetalCommandBufferEncoding {
    type CommandBuffer = MetalCommandBuffer;

    fn context(&self) -> &MetalContext {
        &self.context
    }

    fn allocate_constant(
        &mut self,
        size: usize,
    ) -> Result<<Metal as Backend>::ConstantBuffer, MetalError> {
        let alignment = usize::min(size, 64).next_power_of_two();
        self.constant_allocator.as_mut().unwrap().allocate(size, alignment, |size| self.context.create_buffer_gpu(size))
    }

    fn allocate_scratch(
        &mut self,
        size: usize,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        self.allocation_pool.allocate(size, |size| self.context.create_buffer_gpu(size))
    }

    fn encode_copy(
        &mut self,
        src: impl BufferRef<Backend = Metal>,
        dst: impl BufferMut<Backend = Metal>,
    ) {
        let size = src.size();
        assert_eq!(size, dst.size());
        if size == 0 {
            return;
        }
        let (src_buffer, src_range) = src.parts();
        let (dst_buffer, dst_range) = dst.as_ref().parts();
        let alignment = ((src_buffer.gpu_address() + src_range.start as u64)
            | (dst_buffer.gpu_address() + dst_range.start as u64))
            & 15;
        let (variant, alignment) = match alignment {
            0 => (0, 0),
            4 | 8 | 12 => (1, 4),
            _ => (2, 1),
        };
        let context = self.context.clone();
        let kernel = context.blit_copy[variant]
            .get_or_init(|| BlitCopyMetalKernel::new(&context, alignment).expect("Failed to create blit copy kernel"));
        kernel.encode(src, dst, size as u64, self);
    }

    fn encode_fill(
        &mut self,
        dst: impl BufferMut<Backend = Metal>,
        value: u8,
    ) {
        let (buffer, range) = dst.as_ref().parts();
        assert!(range.end > range.start);
        assert!(range.start.is_multiple_of(4) && range.end.is_multiple_of(4));
        let aligned = (buffer.gpu_address() + range.start as u64).is_multiple_of(16);
        let context = self.context.clone();
        let kernel = context.blit_fill[usize::from(aligned)]
            .get_or_init(|| BlitFillMetalKernel::new(&context, aligned).expect("Failed to create blit fill kernel"));
        kernel.encode(dst, (range.end - range.start) as u64, u32::from(value) * 0x0101_0101, self);
    }

    fn debug_group(
        &mut self,
        name: &str,
    ) -> MetalDebugGroupSpan<'_> {
        ProtocolObject::<dyn MTL4CommandEncoder>::push_debug_group(self.compute_encoder.as_ref(), name);
        self.debug_groups.push(name);
        if let Some(timestamps) = &mut self.timestamps {
            timestamps.start(name, &self.context.device, &self.compute_encoder);
        }
        MetalDebugGroupSpan {
            command_buffer: self,
        }
    }

    fn debug_point(
        &mut self,
        name: &str,
    ) {
        ProtocolObject::<dyn MTL4CommandEncoder>::insert_debug_signpost(self.compute_encoder.as_ref(), name);
    }

    fn with_timestamps(mut self) -> Self {
        let mut timestamps = MetalTimestampRecorder::default();
        if let Some(label) = &self.label {
            timestamps.start(label, &self.context.device, &self.compute_encoder);
        }
        self.timestamps = Some(timestamps);
        self
    }

    fn end_encoding(mut self) -> <Self::CommandBuffer as CommandBuffer>::Executable {
        if self.label.is_some()
            && let Some(timestamps) = &mut self.timestamps
        {
            timestamps.end(&self.context.device, &self.compute_encoder);
        }

        let constant_allocator = self.constant_allocator.take().unwrap();
        assert!(constant_allocator.is_done(), "attempted to end encoding while constants are still alive");

        if let Some(segment) = &self.active {
            self.finish_command_buffer(segment, MTL4VisibilityOptions::Device);
        }
        MetalCommandBufferExecutable {
            segment: self.active.take(),
            resources: Some((constant_allocator, self.allocation_pool.clone())),
            context: self.context.clone(),
            timestamps: self.timestamps.take(),
            segments: self.segments.clone(),
        }
    }
}

impl Drop for MetalCommandBufferEncoding {
    fn drop(&mut self) {
        if let Some(segment) = self.active.take() {
            self.finish_command_buffer(&segment, MTL4VisibilityOptions::Device);
            self.context.command_allocator_cache.lock().push(segment.allocator);
        }
        // Committed command buffers may still run: what they use stays alive until they have all
        // reported.
        if let Some(constant_allocator) = self.constant_allocator.take() {
            self.segments.finish_when_reported((constant_allocator, self.allocation_pool.clone()), |_, _| {});
        }
    }
}

pub struct MetalDebugGroupSpan<'a> {
    command_buffer: &'a mut MetalCommandBufferEncoding,
}

impl Deref for MetalDebugGroupSpan<'_> {
    type Target = MetalCommandBufferEncoding;

    fn deref(&self) -> &Self::Target {
        self.command_buffer
    }
}

impl DerefMut for MetalDebugGroupSpan<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.command_buffer
    }
}

impl Drop for MetalDebugGroupSpan<'_> {
    fn drop(&mut self) {
        let command_buffer = &mut *self.command_buffer;
        if command_buffer.active.is_some() {
            if let Some(timestamps) = &mut command_buffer.timestamps {
                timestamps.end(&command_buffer.context.device, &command_buffer.compute_encoder);
            }
            command_buffer.compute_encoder.pop_debug_group();
        }
        command_buffer.debug_groups.pop();
    }
}

pub struct MetalCommandBufferExecutable {
    /// The segment still to commit, unless starting it failed.
    segment: Option<Segment>,
    /// What the segments use, until they have all reported.
    resources: Option<SegmentResources>,
    context: Arc<MetalContext>,
    timestamps: Option<MetalTimestampRecorder>,
    segments: Arc<Segments>,
}

impl Drop for MetalCommandBufferExecutable {
    fn drop(&mut self) {
        // Dropped without a submit: segments committed earlier may still run.
        if let Some(segment) = self.segment.take() {
            self.context.command_allocator_cache.lock().push(segment.allocator);
        }
        if let Some(resources) = self.resources.take() {
            self.segments.finish_when_reported(resources, |_, _| {});
        }
    }
}

impl CommandBufferExecutable for MetalCommandBufferExecutable {
    type CommandBuffer = MetalCommandBuffer;

    fn submit(mut self) -> MetalCommandBufferPending {
        if let Some(segment) = self.segment.take() {
            self.segments.commit(&self.context, segment);
        }

        // The encoding completes with the last segment to report.
        let (sender, receiver) = mpsc::unbounded();
        let resources = self.resources.take().unwrap();
        let context = self.context.clone();
        let timestamps = self.timestamps.take();
        self.segments.finish_when_reported(resources, move |error, gpu_time| {
            let message = if let Some(error) = error {
                Err(MetalError::CommandBufferExecution(error))
            } else {
                timestamps
                    .as_ref()
                    .map_or_else(|| Ok(Box::default()), |timestamps| timestamps.resolve(context.gpu_clock()))
                    .map(|spans| (gpu_time, spans))
            };
            let _ = sender.unbounded_send(message);
        });

        MetalCommandBufferPending {
            receiver,
        }
    }
}

pub struct MetalCommandBufferPending {
    receiver: mpsc::UnboundedReceiver<Result<(Duration, Box<[TimestampSpan]>), MetalError>>,
}

impl CommandBufferPending for MetalCommandBufferPending {
    type CommandBuffer = MetalCommandBuffer;

    async fn wait_until_completed(mut self) -> Result<MetalCommandBufferCompleted, MetalError> {
        let start = std::time::Instant::now();
        let (gpu_execution_time, timestamps) = loop {
            if start.elapsed() >= Duration::from_millis(50) {
                break self.receiver.recv().await.map_err(|_| MetalError::CommandBufferWait)?;
            }
            match self.receiver.try_recv() {
                Ok(message) => break message,
                Err(mpsc::TryRecvError::Empty) => std::thread::sleep(Duration::from_micros(20)),
                Err(mpsc::TryRecvError::Closed) => return Err(MetalError::CommandBufferWait),
            }
        }?;
        Ok(MetalCommandBufferCompleted {
            gpu_execution_time,
            timestamps,
        })
    }
}

pub struct MetalCommandBufferCompleted {
    gpu_execution_time: Duration,
    timestamps: Box<[TimestampSpan]>,
}

impl CommandBufferCompleted for MetalCommandBufferCompleted {
    type CommandBuffer = MetalCommandBuffer;

    fn gpu_execution_time(&self) -> Option<Duration> {
        Some(self.gpu_execution_time)
    }

    fn timestamps(&self) -> &[TimestampSpan] {
        &self.timestamps
    }
}

#[cfg(test)]
#[path = "../../../unit/backends/metal/command_buffer_handoff_test.rs"]
mod handoff_tests;
