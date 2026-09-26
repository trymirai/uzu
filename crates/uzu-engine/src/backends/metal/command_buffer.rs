use std::{
    range::Range,
    sync::{Arc, mpsc},
    time::Duration,
};

use metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer, MTL4CommandEncoder,
    MTL4CommandEncoderExt, MTL4CommandQueueExt, MTL4CommitFeedback, MTL4CommitFeedbackExt, MTL4CommitFeedbackHandler,
    MTL4CommitOptions, MTL4ComputeCommandEncoder, MTL4ComputeCommandEncoderExt, MTL4VisibilityOptions, MTLDeviceExt,
    MTLStages,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use rangemap::RangeSet;

use crate::backends::{
    common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding,
        CommandBufferExecutable, CommandBufferPending, allocator::AllocationType,
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
    pub(super) context: Arc<MetalContext>,
    allocation_pool: Arc<<Metal as Backend>::AllocationPool>,
}

impl MetalCommandBufferEncoding {
    pub fn new(
        command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
        command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
        context: Arc<MetalContext>,
        allocation_pool: Arc<<Metal as Backend>::AllocationPool>,
    ) -> Result<Self, MetalError> {
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

        Ok(Self {
            command_allocator,
            command_buffer,
            compute_encoder,
            argument_table,
            reads: RangeSet::new(),
            writes: RangeSet::new(),
            context,
            allocation_pool,
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
        self.context.allocator.allocate(
            size,
            AllocationType::Pooled {
                pool: &self.allocation_pool,
                cpu_available: true,
            },
        )
    }

    fn allocate_scratch(
        &mut self,
        size: usize,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        self.context.allocator.allocate(
            size,
            AllocationType::Pooled {
                pool: &self.allocation_pool,
                cpu_available: false,
            },
        )
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

    fn end_encoding(self) -> <Self::CommandBuffer as CommandBuffer>::Executable {
        MetalCommandBufferExecutable {
            command_allocator: self.command_allocator.clone(),
            command_buffer: self.command_buffer.clone(),
            context: self.context.clone(),
            allocation_pool: self.allocation_pool.clone(),
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
    context: Arc<MetalContext>,
    allocation_pool: Arc<<Metal as Backend>::AllocationPool>,
}

impl CommandBufferExecutable for MetalCommandBufferExecutable {
    type CommandBuffer = MetalCommandBuffer;

    fn submit(self) -> MetalCommandBufferPending {
        let (sender, receiver) = mpsc::channel();

        let feedback_handler = move |feedback: &ProtocolObject<dyn MTL4CommitFeedback>| {
            let message = if let Some(error) = feedback.error() {
                Err(error.to_string())
            } else {
                Ok(Duration::from_secs_f64(feedback.gpu_end_time() - feedback.gpu_start_time()))
            };
            let _ = sender.send(message);
        };

        let options = MTL4CommitOptions::new();
        options.add_feedback_handler(&MTL4CommitFeedbackHandler::new(feedback_handler));
        self.context.command_queue.commit_with_options(&[&self.command_buffer], &options);

        MetalCommandBufferPending {
            _command_allocator: self.command_allocator,
            _command_buffer: self.command_buffer,
            allocation_pool: self.allocation_pool,
            receiver,
        }
    }
}

pub struct MetalCommandBufferPending {
    _command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    _command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    allocation_pool: Arc<<Metal as Backend>::AllocationPool>,
    receiver: mpsc::Receiver<Result<Duration, String>>,
}

impl CommandBufferPending for MetalCommandBufferPending {
    type CommandBuffer = MetalCommandBuffer;

    fn wait_until_completed(self) -> Result<MetalCommandBufferCompleted, MetalError> {
        Ok(MetalCommandBufferCompleted {
            gpu_execution_time: self
                .receiver
                .recv_timeout(Duration::from_secs(60))
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
