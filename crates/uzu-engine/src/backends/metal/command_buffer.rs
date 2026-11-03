use std::{
    collections::HashMap,
    mem,
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
    MTL4VisibilityOptions, MTLDeviceExt, MTLEvent, MTLSharedEvent, MTLStages,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use parking_lot::Mutex;
use rangemap::RangeSet;

use crate::backends::{
    common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding,
        CommandBufferExecutable, CommandBufferPending, Context, TimestampSpan, allocator::bump::BumpAllocator,
    },
    metal::{
        Metal,
        buffer::MetalBufferExt,
        context::{MetalContext, MetalLane},
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

pub struct MetalCommandBufferEncoding {
    command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    pub(super) compute_encoder: Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>,
    pub(super) argument_table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    reads: RangeSet<u64>,
    writes: RangeSet<u64>,
    constant_allocator: Option<BumpAllocator<<Metal as Backend>::GlobalGpuBuffer>>,
    allocation_pool: Arc<<Metal as Backend>::AllocationPool>,
    pub(super) context: Arc<MetalContext>,
    timestamps: Option<MetalTimestampRecorder>,
    debug_groups: Vec<String>,
    handoffs: Option<Arc<Handoffs>>,
    /// The queue (of lane `lane`) the current segment is committed to.
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    lane: usize,
    /// Both lanes, once [`Self::begin_independent_chunk`] switched lanes.
    lanes: Option<[Arc<MetalLane>; 2]>,
    /// The scratch pool of the lane not in use; scratch reuse is only ordered within one queue.
    other_pool: Option<Arc<<Metal as Backend>::AllocationPool>>,
    /// Events the queue waits for before the next commit.
    pending_waits: Vec<EventWait>,
    /// Layers' progress of the chunk being encoded (see [`Self::record_layer_ready`]).
    layer_ready: HashMap<usize, u64>,
    /// The chunk before the one being encoded, which runs on the other lane.
    previous_chunk: Option<PreviousChunk>,
}

unsafe impl Send for MetalCommandBufferEncoding {}

type EventWait = (Retained<ProtocolObject<dyn MTLEvent>>, u64);

/// What a chunk's layers wait for in the chunk before it.
struct PreviousChunk {
    /// The previous chunk's lane's order event.
    order: Retained<ProtocolObject<dyn MTLEvent>>,
    /// Reached once everything the chunk encoded has completed.
    tail: u64,
    /// Per layer, the value reached once the chunk's work up to the layer's MLP input has completed.
    layer_ready: HashMap<usize, u64>,
}

/// State shared by the command buffers an encoding was cut into by [`MetalCommandBufferEncoding::handoff`].
#[derive(Default)]
pub struct Handoffs {
    first_gpu_start: Mutex<Option<f64>>,
    error: Mutex<Option<String>>,
}

impl Handoffs {
    /// Fails the execution of the encoding: awaiting it returns this error. Must be called before
    /// the handed-off work is reported done to the queue, so the final command buffer sees it.
    pub fn fail(
        &self,
        message: &str,
    ) {
        self.error.lock().get_or_insert_with(|| message.to_string());
    }
}

impl MetalCommandBufferEncoding {
    pub fn new(
        context: Arc<MetalContext>,
        name: Option<&str>,
        allocation_pool: Option<Arc<<Metal as Backend>::AllocationPool>>,
    ) -> Result<Self, MetalError> {
        let command_allocator = new_command_allocator(&context)?;

        let command_buffer = context.device.new_mtl4_command_buffer().ok_or(MetalError::CannotCreateCommandBuffer)?;
        command_buffer.begin_command_buffer_with_allocator(&command_allocator);
        command_buffer.set_label(name);

        let argument_table_descriptor = MTL4ArgumentTableDescriptor::new();
        argument_table_descriptor.set_label(name.map(|name| format!("{name} argument table")).as_deref());
        argument_table_descriptor.set_max_buffer_bind_count(31);
        let argument_table = context
            .device
            .new_argument_table_with_descriptor(&argument_table_descriptor)
            .map_err(|error| MetalError::CannotCreateArgumentTable(error.to_string()))?;
        let compute_encoder = begin_compute_encoder(&command_buffer, &argument_table, name);

        let constant_allocator = BumpAllocator::new(256 * 1024, 1);

        let allocation_pool = allocation_pool.unwrap_or_else(|| context.create_allocation_pool());

        Ok(Self {
            command_allocator,
            command_buffer,
            compute_encoder,
            argument_table,
            reads: RangeSet::new(),
            writes: RangeSet::new(),
            constant_allocator: Some(constant_allocator),
            allocation_pool,
            timestamps: None,
            debug_groups: Vec::new(),
            handoffs: None,
            queue: context.command_queue.clone(),
            lane: 0,
            lanes: None,
            other_pool: None,
            pending_waits: Vec::new(),
            layer_ready: HashMap::new(),
            previous_chunk: None,
            context,
        })
    }

    /// Hands the GPU timeline over to another engine at this point of the encoding.
    ///
    /// Everything encoded so far is submitted; the queue then signals `event` to `signal` once that
    /// work completes and holds everything encoded afterwards until `event` reaches `wait`. Encoding
    /// continues transparently (debug groups and timestamps carry over). An encoding that never hands
    /// off is submitted exactly as before.
    pub fn handoff(
        &mut self,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        signal: Option<u64>,
        wait: Option<u64>,
    ) -> Result<(), MetalError> {
        self.commit_segment();

        let event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(event);
        if let Some(signal) = signal {
            self.queue.signal_event_value(event, signal);
        }
        if let Some(wait) = wait {
            self.queue.wait_for_event_value(event, wait);
        }

        self.begin_segment()
    }

    /// [`Self::handoff`] for an ANE round trip on `lane` (the encoding's current lane): the ANE
    /// waits for `lane.event` to reach `ready`, and the queue then waits for `lane.done` to reach
    /// `done` (the ANE executor relays the request's completion to it). `lane.order` also reaches
    /// `ready`, as soon as the work so far completes, for the next chunk's GPU work to order against.
    pub fn handoff_to_ane(
        &mut self,
        lane: &MetalLane,
        ready: u64,
        done: u64,
    ) -> Result<(), MetalError> {
        self.handoff(&lane.event, Some(ready), None)?;
        self.queue.signal_event_value(&lane.order, ready);
        self.queue.wait_for_event_value(&lane.done, done);
        Ok(())
    }

    /// Where work handed off to another engine reports its failure: a failure recorded here makes
    /// awaiting the encoding fail, instead of the encoding completing with a silently wrong result.
    pub fn failure_sink(&mut self) -> Arc<Handoffs> {
        self.handoffs.get_or_insert_with(Arc::default).clone()
    }

    /// The lane (command queue) the encoding is currently on: 0 until a chunk was pipelined.
    pub fn lane(&self) -> usize {
        self.lane
    }

    /// Holds the segment being encoded back until the previous chunk has passed `layer`'s MLP
    /// input (or has completed, if it never recorded the layer). A chunk's layer-L mixer reads
    /// and updates state the previous chunk's layer-L mixer wrote.
    pub fn order_after_previous_chunk(
        &mut self,
        layer: usize,
    ) {
        if let Some(previous) = &self.previous_chunk {
            let value = previous.layer_ready.get(&layer).copied().unwrap_or(previous.tail);
            self.pending_waits.push((previous.order.clone(), value));
        }
    }

    /// Records that this chunk's work up to `layer`'s MLP input completes when the current
    /// lane's event reaches `value`, for the next chunk to order against.
    pub fn record_layer_ready(
        &mut self,
        layer: usize,
        value: u64,
    ) {
        self.layer_ready.insert(layer, value);
    }

    /// Ends the current segment: queue waits requested so far, then the commit.
    fn commit_segment(&mut self) {
        let handoffs = self.handoffs.get_or_insert_with(Arc::default).clone();

        for _ in &self.debug_groups {
            self.compute_encoder.pop_debug_group();
        }
        self.finish_command_buffer();
        sync_sparse_queue(&self.context, &self.queue, self.lane);
        for (event, value) in self.pending_waits.drain(..) {
            self.queue.wait_for_event_value(&event, value);
        }

        let command_allocator = self.command_allocator.clone();
        let command_buffer = self.command_buffer.clone();
        let context = self.context.clone();
        let feedback_handler = move |feedback: &ProtocolObject<dyn MTL4CommitFeedback>| {
            if let Some(error) = feedback.error() {
                handoffs.error.lock().get_or_insert(error.to_string());
            }
            handoffs.first_gpu_start.lock().get_or_insert(feedback.gpu_start_time());
            context.command_allocator_cache.lock().push(command_allocator.clone());
            let _keep_alive = &command_buffer;
        };
        let options = MTL4CommitOptions::new();
        options.add_feedback_handler(&MTL4CommitFeedbackHandler::new(feedback_handler));
        self.queue.commit_with_options(&[&self.command_buffer], &options);
    }

    /// Starts a fresh command buffer on the current queue (debug groups carry over).
    fn begin_segment(&mut self) -> Result<(), MetalError> {
        self.command_allocator = new_command_allocator(&self.context)?;
        self.command_buffer =
            self.context.device.new_mtl4_command_buffer().ok_or(MetalError::CannotCreateCommandBuffer)?;
        self.command_buffer.begin_command_buffer_with_allocator(&self.command_allocator);
        self.compute_encoder = begin_compute_encoder(&self.command_buffer, &self.argument_table, None);
        self.reads.clear();
        self.writes.clear();
        for name in &self.debug_groups {
            ProtocolObject::<dyn MTL4CommandEncoder>::push_debug_group(self.compute_encoder.as_ref(), name);
        }
        Ok(())
    }

    fn finish_command_buffer(&mut self) {
        self.compute_encoder.barrier_after_stages_before_queue_stages_visibility_options(
            MTLStages::Dispatch,
            MTLStages::Dispatch | MTLStages::ResourceState,
            MTL4VisibilityOptions::Device,
        );
        self.compute_encoder.end_encoding();
        self.command_buffer.end_command_buffer();
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

fn new_command_allocator(
    context: &MetalContext
) -> Result<Retained<ProtocolObject<dyn MTL4CommandAllocator>>, MetalError> {
    if let Some(command_allocator) = context.command_allocator_cache.lock().pop() {
        command_allocator.reset();
        Ok(command_allocator)
    } else {
        context.device.new_command_allocator().ok_or(MetalError::CannotCreateCommandBuffer)
    }
}

fn begin_compute_encoder(
    command_buffer: &ProtocolObject<dyn MTL4CommandBuffer>,
    argument_table: &ProtocolObject<dyn MTL4ArgumentTable>,
    name: Option<&str>,
) -> Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>> {
    let compute_encoder = command_buffer.compute_command_encoder().unwrap();
    ProtocolObject::<dyn MTL4CommandEncoder>::set_label(compute_encoder.as_ref(), name);
    compute_encoder.barrier_after_queue_stages_before_stages_visibility_options(
        MTLStages::Dispatch | MTLStages::ResourceState,
        MTLStages::Dispatch,
        MTL4VisibilityOptions::Device,
    );
    compute_encoder.set_argument_table(Some(argument_table));
    compute_encoder
}

/// Makes `queue` (of lane `lane`) wait for the sparse mappings made since it last did.
fn sync_sparse_queue(
    context: &MetalContext,
    queue: &ProtocolObject<dyn MTL4CommandQueue>,
    lane: usize,
) {
    let mut sparse_state_locked = context.sparse_state.lock();
    if sparse_state_locked.did_ops {
        sparse_state_locked.did_ops = false;
        sparse_state_locked.event_value += 1;
        let value = sparse_state_locked.event_value;
        context.sparse_queue.signal_event_value(&context.sparse_event, value);
    }
    let value = sparse_state_locked.event_value;
    if sparse_state_locked.synced[lane] < value {
        sparse_state_locked.synced[lane] = value;
        queue.wait_for_event_value(&context.sparse_event, value);
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
        self.debug_groups.push(name.to_owned());
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

    /// Moves encoding to the other lane: the chunk encoded so far keeps running on its lane while
    /// the next one starts on the other. Only the ANE MLP executors enforce the per-layer ordering
    /// that makes this safe, so without one the encoding stays on a single lane.
    fn begin_independent_chunk(&mut self) -> Result<(), MetalError> {
        if !self.context.ane_pipelining_available() {
            return Ok(());
        }
        let lanes = [self.context.lane(0)?, self.context.lane(1)?];
        let current = lanes[self.lane].clone();
        let next = lanes[1 - self.lane].clone();
        self.lanes.get_or_insert(lanes);

        self.commit_segment();
        let tail = current.reserve_values(1);
        self.queue.signal_event_value(&current.order, tail);
        self.previous_chunk = Some(PreviousChunk {
            order: current.order.clone(),
            tail,
            layer_ready: mem::take(&mut self.layer_ready),
        });

        self.lane = 1 - self.lane;
        self.queue = next.queue.clone();
        let next_pool = self.other_pool.take().unwrap_or_else(|| self.context.create_allocation_pool());
        self.other_pool = Some(mem::replace(&mut self.allocation_pool, next_pool));
        self.begin_segment()
    }

    fn with_timestamps(mut self) -> Self {
        self.timestamps = Some(MetalTimestampRecorder::default());
        self
    }

    fn end_encoding(mut self) -> <Self::CommandBuffer as CommandBuffer>::Executable {
        let constant_allocator = self.constant_allocator.take().unwrap();
        assert!(constant_allocator.is_done(), "attempted to end encoding while constants are still alive");

        // Completing the final command buffer must imply that every lane completed (scratch pools
        // and constants are released then), so it waits for the other lane's last chunk.
        let mut waits = mem::take(&mut self.pending_waits);
        if let Some(previous) = &self.previous_chunk {
            waits.push((previous.order.clone(), previous.tail));
        }
        // Later encodings go to the main queue; hold it back until this one has finished.
        let main_queue_release = if self.lane != 0 {
            self.lanes.as_ref().map(|lanes| lanes[self.lane].clone())
        } else {
            None
        };

        MetalCommandBufferExecutable {
            command_allocator: self.command_allocator.clone(),
            command_buffer: self.command_buffer.clone(),
            constant_allocator,
            allocation_pool: self.allocation_pool.clone(),
            other_pool: self.other_pool.take(),
            context: self.context.clone(),
            timestamps: self.timestamps.take(),
            handoffs: self.handoffs.take(),
            queue: self.queue.clone(),
            lane: self.lane,
            waits,
            main_queue_release,
        }
    }
}

impl Drop for MetalCommandBufferEncoding {
    fn drop(&mut self) {
        self.finish_command_buffer();
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
        if let Some(timestamps) = &mut command_buffer.timestamps {
            timestamps.end(&command_buffer.context.device, &command_buffer.compute_encoder);
        }
        command_buffer.debug_groups.pop();
        command_buffer.compute_encoder.pop_debug_group();
    }
}

pub struct MetalCommandBufferExecutable {
    command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    constant_allocator: BumpAllocator<<Metal as Backend>::GlobalGpuBuffer>,
    allocation_pool: Arc<<Metal as Backend>::AllocationPool>,
    other_pool: Option<Arc<<Metal as Backend>::AllocationPool>>,
    context: Arc<MetalContext>,
    timestamps: Option<MetalTimestampRecorder>,
    handoffs: Option<Arc<Handoffs>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    lane: usize,
    waits: Vec<EventWait>,
    main_queue_release: Option<Arc<MetalLane>>,
}

impl CommandBufferExecutable for MetalCommandBufferExecutable {
    type CommandBuffer = MetalCommandBuffer;

    fn submit(self) -> MetalCommandBufferPending {
        sync_sparse_queue(&self.context, &self.queue, self.lane);
        for (event, value) in &self.waits {
            self.queue.wait_for_event_value(event, *value);
        }

        let (sender, receiver) = mpsc::unbounded();

        let command_allocator = self.command_allocator.clone();
        let command_buffer = self.command_buffer.clone();
        let context_clone = self.context.clone();
        let timestamps = self.timestamps;

        let constant_allocator = self.constant_allocator;
        let allocation_pool = self.allocation_pool.clone();
        let other_pool = self.other_pool.clone();
        let handoffs = self.handoffs;
        let feedback_handler = move |feedback: &ProtocolObject<dyn MTL4CommitFeedback>| {
            let mut error = feedback.error().map(|error| error.to_string());
            let mut gpu_start_time = feedback.gpu_start_time();
            if let Some(handoffs) = &handoffs {
                error = error.or_else(|| handoffs.error.lock().take());
                if let Some(first_gpu_start) = *handoffs.first_gpu_start.lock() {
                    gpu_start_time = first_gpu_start;
                }
            }
            let message = if let Some(error) = error {
                Err(MetalError::CommandBufferExecution(error))
            } else {
                timestamps
                    .as_ref()
                    .map_or_else(|| Ok(Box::default()), |timestamps| timestamps.resolve(&context_clone.device))
                    .map(|spans| (Duration::from_secs_f64(feedback.gpu_end_time() - gpu_start_time), spans))
            };
            let _ = sender.unbounded_send(message);
            context_clone.command_allocator_cache.lock().push(command_allocator.clone());
            let _keep_alive = (&command_buffer, &constant_allocator, &allocation_pool, &other_pool);
        };

        let options = MTL4CommitOptions::new();
        options.add_feedback_handler(&MTL4CommitFeedbackHandler::new(feedback_handler));
        self.queue.commit_with_options(&[&self.command_buffer], &options);
        if let Some(lane) = &self.main_queue_release {
            let value = lane.reserve_values(1);
            self.queue.signal_event_value(&lane.order, value);
            self.context.command_queue.wait_for_event_value(&lane.order, value);
        }

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
        let (gpu_execution_time, timestamps) =
            self.receiver.recv().await.map_err(|_| MetalError::CommandBufferWait)??;
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
