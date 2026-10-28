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
    MTL4VisibilityOptions, MTLDeviceExt, MTLStages,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use rangemap::RangeSet;

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
}

unsafe impl Send for MetalCommandBufferEncoding {}

impl MetalCommandBufferEncoding {
    pub fn new(
        context: Arc<MetalContext>,
        name: Option<&str>,
        allocation_pool: Option<Arc<<Metal as Backend>::AllocationPool>>,
    ) -> Result<Self, MetalError> {
        let command_allocator = if let Some(command_allocator) = context.command_allocator_cache.lock().pop() {
            command_allocator.reset();
            command_allocator
        } else {
            context.device.new_command_allocator().ok_or(MetalError::CannotCreateCommandBuffer)?
        };

        let command_buffer = context.device.new_mtl4_command_buffer().ok_or(MetalError::CannotCreateCommandBuffer)?;
        command_buffer.begin_command_buffer_with_allocator(&command_allocator);
        command_buffer.set_label(name);

        let compute_encoder = command_buffer.compute_command_encoder().unwrap();
        ProtocolObject::<dyn MTL4CommandEncoder>::set_label(compute_encoder.as_ref(), name);

        compute_encoder.barrier_after_queue_stages_before_stages_visibility_options(
            MTLStages::Dispatch | MTLStages::ResourceState,
            MTLStages::Dispatch,
            MTL4VisibilityOptions::Device,
        );

        let argument_table_descriptor = MTL4ArgumentTableDescriptor::new();
        argument_table_descriptor.set_label(name.map(|name| format!("{name} argument table")).as_deref());
        argument_table_descriptor.set_max_buffer_bind_count(31);
        let argument_table = context
            .device
            .new_argument_table_with_descriptor(&argument_table_descriptor)
            .map_err(|error| MetalError::CannotCreateArgumentTable(error.to_string()))?;
        compute_encoder.set_argument_table(Some(&argument_table));

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
            context,
            timestamps: None,
        })
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
        self.timestamps = Some(MetalTimestampRecorder::default());
        self
    }

    fn end_encoding(mut self) -> <Self::CommandBuffer as CommandBuffer>::Executable {
        let constant_allocator = self.constant_allocator.take().unwrap();
        assert!(constant_allocator.is_done(), "attempted to end encoding while constants are still alive");

        MetalCommandBufferExecutable {
            command_allocator: self.command_allocator.clone(),
            command_buffer: self.command_buffer.clone(),
            constant_allocator,
            allocation_pool: self.allocation_pool.clone(),
            context: self.context.clone(),
            timestamps: self.timestamps.take(),
        }
    }
}

impl Drop for MetalCommandBufferEncoding {
    fn drop(&mut self) {
        self.compute_encoder.barrier_after_stages_before_queue_stages_visibility_options(
            MTLStages::Dispatch,
            MTLStages::Dispatch | MTLStages::ResourceState,
            MTL4VisibilityOptions::Device,
        );
        self.compute_encoder.end_encoding();
        self.command_buffer.end_command_buffer();
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
        command_buffer.compute_encoder.pop_debug_group();
    }
}

pub struct MetalCommandBufferExecutable {
    command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    constant_allocator: BumpAllocator<<Metal as Backend>::GlobalGpuBuffer>,
    allocation_pool: Arc<<Metal as Backend>::AllocationPool>,
    context: Arc<MetalContext>,
    timestamps: Option<MetalTimestampRecorder>,
}

impl CommandBufferExecutable for MetalCommandBufferExecutable {
    type CommandBuffer = MetalCommandBuffer;

    fn submit(self) -> MetalCommandBufferPending {
        let mut sparse_state_locked = self.context.sparse_state.lock();
        if sparse_state_locked.did_ops {
            sparse_state_locked.did_ops = false;
            sparse_state_locked.event_value += 1;
            let value = sparse_state_locked.event_value;
            self.context.sparse_queue.signal_event_value(&self.context.sparse_event, value);
            self.context.command_queue.wait_for_event_value(&self.context.sparse_event, value);
        }
        drop(sparse_state_locked);

        let (sender, receiver) = mpsc::unbounded();

        let command_allocator = self.command_allocator.clone();
        let command_buffer = self.command_buffer.clone();
        let context_clone = self.context.clone();
        let timestamps = self.timestamps;

        let constant_allocator = self.constant_allocator;
        let allocation_pool = self.allocation_pool.clone();
        let feedback_handler = move |feedback: &ProtocolObject<dyn MTL4CommitFeedback>| {
            let message = if let Some(error) = feedback.error() {
                Err(MetalError::CommandBufferExecution(error.to_string()))
            } else {
                timestamps
                    .as_ref()
                    .map_or_else(|| Ok(Box::default()), |timestamps| timestamps.resolve(&context_clone.device))
                    .map(|spans| (Duration::from_secs_f64(feedback.gpu_end_time() - feedback.gpu_start_time()), spans))
            };
            let _ = sender.unbounded_send(message);
            context_clone.command_allocator_cache.lock().push(command_allocator.clone());
            let _keep_alive = (&command_buffer, &constant_allocator, &allocation_pool);
        };

        let options = MTL4CommitOptions::new();
        options.add_feedback_handler(&MTL4CommitFeedbackHandler::new(feedback_handler));
        self.context.command_queue.commit_with_options(&[&self.command_buffer], &options);

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
