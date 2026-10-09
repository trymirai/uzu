use std::{
    range::Range,
    sync::{Arc, mpsc},
    time::Duration,
};

use metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer, MTL4CommandBufferExt,
    MTL4CommandEncoder, MTL4CommandEncoderExt, MTL4CommandQueue, MTL4CommandQueueExt, MTL4CommitFeedback,
    MTL4CommitFeedbackExt, MTL4CommitFeedbackHandler, MTL4CommitOptions, MTL4ComputeCommandEncoder,
    MTL4ComputeCommandEncoderExt, MTL4VisibilityOptions, MTLDeviceExt, MTLStages,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use rangemap::RangeSet;

use crate::backends::{
    common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding,
        CommandBufferExecutable, CommandBufferPending, Context, allocator::bump::BumpAllocator,
    },
    metal::{Metal, MetalContext, buffer::MetalBufferExt, error::MetalError},
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
    constant_allocator: Option<BumpAllocator<<Metal as Backend>::GlobalBuffer>>,
    allocation_pool: Arc<<Metal as Backend>::AllocationPool>,
    pub(super) context: Arc<MetalContext>,
}

impl MetalCommandBufferEncoding {
    pub fn new(
        context: Arc<MetalContext>,
        name: Option<&str>,
        allocation_pool: Option<Arc<<Metal as Backend>::AllocationPool>>,
    ) -> Result<Self, MetalError> {
        let (command_allocator, command_buffer) = if let mut command_buffer_cache = context.command_buffer_cache.lock()
            && let Some(command_buffer_cached) = command_buffer_cache.pop()
        {
            command_buffer_cached.command_allocator.reset();
            (command_buffer_cached.command_allocator, command_buffer_cached.command_buffer)
        } else {
            (
                context.device.new_command_allocator().ok_or(MetalError::CannotCreateCommandBuffer)?,
                context.device.new_mtl4_command_buffer().ok_or(MetalError::CannotCreateCommandBuffer)?,
            )
        };

        command_buffer.set_label(name);

        command_buffer.begin_command_buffer_with_allocator(&command_allocator);

        let compute_encoder = command_buffer.compute_command_encoder().unwrap();

        compute_encoder.barrier_after_queue_stages_before_stages_visibility_options(
            MTLStages::Dispatch | MTLStages::Blit | MTLStages::ResourceState,
            MTLStages::Dispatch | MTLStages::Blit,
            MTL4VisibilityOptions::Device,
        );

        let argument_table_descriptor = MTL4ArgumentTableDescriptor::new();
        argument_table_descriptor.set_max_buffer_bind_count(31);
        let argument_table = context
            .device
            .new_argument_table_with_descriptor(&argument_table_descriptor)
            .map_err(|error| MetalError::CannotCreateArgumentTable(error.to_string()))?;
        compute_encoder.set_argument_table(Some(&argument_table));

        let constant_allocator = BumpAllocator::new(256 * 1024);

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
                MTLStages::Dispatch | MTLStages::Blit,
                MTLStages::Dispatch | MTLStages::Blit,
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
        self.constant_allocator.as_mut().unwrap().allocate(size, |size| self.context.create_buffer(size))
    }

    fn allocate_scratch(
        &mut self,
        size: usize,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        self.allocation_pool.allocate(size, |size| self.context.create_buffer(size))
    }

    fn encode_copy(
        &mut self,
        src: impl BufferRef<Backend = Metal>,
        dst: impl BufferMut<Backend = Metal>,
    ) {
        let (src, src_range) = src.parts();
        let (dst, dst_range) = dst.parts();
        assert_eq!(src_range.iter().len(), dst_range.iter().len());

        self.access(&[
            Access {
                range: src.gpu_address_subrange(src_range),
                write: false,
            },
            Access {
                range: dst.gpu_address_subrange(dst_range),
                write: true,
            },
        ]);

        let (src_buffer, src_offset) = src.downcast();
        let (dst_buffer, dst_offset) = dst.downcast();
        self.compute_encoder.copy_from_buffer_source_offset_to_buffer_destination_offset_size(
            src_buffer,
            src_offset + src_range.start,
            dst_buffer,
            dst_offset + dst_range.start,
            src_range.iter().len(),
        );
    }

    fn encode_fill(
        &mut self,
        dst: impl BufferMut<Backend = Metal>,
        value: u8,
    ) {
        let (dst, range) = dst.parts();
        assert!(range.end > range.start);
        assert!(range.start.is_multiple_of(4) && range.end.is_multiple_of(4));

        self.access(&[Access {
            range: dst.gpu_address_subrange(range),
            write: true,
        }]);

        let (buffer, offset) = dst.downcast();
        self.compute_encoder.fill_buffer_range_value(buffer, offset + range.start..offset + range.end, value);
    }

    // TODO: maybe port previous debug command_buffer labels
    fn push_debug_group(
        &mut self,
        name: &str,
    ) {
        ProtocolObject::<dyn MTL4CommandEncoder>::push_debug_group(self.compute_encoder.as_ref(), name);
    }

    fn pop_debug_group(&mut self) {
        self.compute_encoder.pop_debug_group();
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
        }
    }
}

impl Drop for MetalCommandBufferEncoding {
    fn drop(&mut self) {
        self.compute_encoder.barrier_after_stages_before_queue_stages_visibility_options(
            MTLStages::Dispatch | MTLStages::Blit,
            MTLStages::Dispatch | MTLStages::Blit | MTLStages::ResourceState,
            MTL4VisibilityOptions::Device,
        );
        self.compute_encoder.end_encoding();
        self.command_buffer.end_command_buffer();
    }
}

pub struct MetalCommandBufferExecutable {
    command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    constant_allocator: BumpAllocator<<Metal as Backend>::GlobalBuffer>,
    allocation_pool: Arc<<Metal as Backend>::AllocationPool>,
    context: Arc<MetalContext>,
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

        let (sender, receiver) = mpsc::channel();

        let command_allocator = self.command_allocator.clone();
        let command_buffer = self.command_buffer.clone();
        let context_clone = self.context.clone();

        let constant_allocator = self.constant_allocator;
        let allocation_pool = self.allocation_pool.clone();
        let feedback_handler = move |feedback: &ProtocolObject<dyn MTL4CommitFeedback>| {
            let message = if let Some(error) = feedback.error() {
                Err(error.to_string())
            } else {
                Ok(Duration::from_secs_f64(feedback.gpu_end_time() - feedback.gpu_start_time()))
            };
            let _ = sender.send(message);
            context_clone.command_buffer_cache.lock().push(MetalCommandBufferCache {
                command_allocator: command_allocator.clone(),
                command_buffer: command_buffer.clone(),
            });
            let _keep_alive = (&constant_allocator, &allocation_pool);
        };

        let options = MTL4CommitOptions::new();
        options.add_feedback_handler(&MTL4CommitFeedbackHandler::new(feedback_handler));
        self.context.command_queue.commit_with_options(&[&self.command_buffer], &options);

        MetalCommandBufferPending {
            allocation_pool: self.allocation_pool,
            receiver,
        }
    }
}

pub struct MetalCommandBufferPending {
    allocation_pool: Arc<<Metal as Backend>::AllocationPool>,
    receiver: mpsc::Receiver<Result<Duration, String>>,
}

impl CommandBufferPending for MetalCommandBufferPending {
    type CommandBuffer = MetalCommandBuffer;

    fn wait_until_completed(self) -> Result<MetalCommandBufferCompleted, MetalError> {
        Ok(MetalCommandBufferCompleted {
            gpu_execution_time: self
                .receiver
                .recv()
                .map_err(MetalError::CommandBufferWait)?
                .map_err(MetalError::CommandBufferExecution)?,
            _allocation_pool: self.allocation_pool,
        })
    }
}

pub struct MetalCommandBufferCompleted {
    gpu_execution_time: Duration,
    _allocation_pool: Arc<<Metal as Backend>::AllocationPool>,
}

impl CommandBufferCompleted for MetalCommandBufferCompleted {
    type CommandBuffer = MetalCommandBuffer;

    fn gpu_execution_time(&self) -> Duration {
        self.gpu_execution_time
    }
}

pub struct MetalCommandBufferCache {
    command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
}
