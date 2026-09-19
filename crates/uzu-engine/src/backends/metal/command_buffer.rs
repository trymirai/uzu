use std::{
    ops::Range,
    sync::{Arc, mpsc},
    time::Duration,
};

use metal::{
    MTL4CommandAllocator, MTL4CommandBuffer, MTL4CommandEncoder, MTL4CommandEncoderExt, MTL4CommandQueueExt,
    MTL4CommitFeedback, MTL4CommitFeedbackExt, MTL4CommitFeedbackHandler, MTL4CommitOptions, MTL4ComputeCommandEncoder,
    MTL4ComputeCommandEncoderExt, MTL4VisibilityOptions, MTLStages,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use rangemap::RangeSet;

use crate::backends::{
    common::{
        Allocation, AllocationPool, AllocationType, Buffer, BufferGpuAddressRangeExt, BufferRangeMut, BufferRangeRef,
        CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
    },
    metal::{Metal, MetalContext, error::MetalError},
};

pub struct MetalCommandBuffer;

impl CommandBuffer for MetalCommandBuffer {
    type Backend = Metal;

    type Encoding = MetalCommandBufferEncoding;
    type Executable = MetalCommandBufferExecutable;
    type Pending = MetalCommandBufferPending;
    type Completed = MetalCommandBufferCompleted;
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct AccessFlags {
    pub(super) read: bool,
    pub(super) write: bool,
}

impl AccessFlags {
    pub(super) fn read() -> Self {
        Self {
            read: true,
            write: false,
        }
    }

    pub(super) fn write() -> Self {
        Self {
            read: false,
            write: true,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct Access {
    pub(super) range: Range<usize>,
    pub(super) flags: AccessFlags,
}

pub struct MetalCommandBufferEncoding {
    command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    pub(super) compute_encoder: Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>,
    reads: RangeSet<usize>,
    writes: RangeSet<usize>,
    pub(super) context: Arc<MetalContext>,
    allocation_pool: Arc<AllocationPool<Metal>>,
}

impl MetalCommandBufferEncoding {
    pub fn new(
        command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
        command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
        context: Arc<MetalContext>,
        allocation_pool: Arc<AllocationPool<Metal>>,
    ) -> Self {
        command_buffer.begin_command_buffer_with_allocator(&command_allocator);

        let compute_encoder = command_buffer.compute_command_encoder().unwrap();

        compute_encoder.barrier_after_queue_stages_before_stages_visibility_options(
            MTLStages::Dispatch | MTLStages::Blit | MTLStages::ResourceState,
            MTLStages::Dispatch | MTLStages::Blit,
            MTL4VisibilityOptions::Device,
        );

        Self {
            command_allocator,
            command_buffer,
            compute_encoder,
            reads: RangeSet::new(),
            writes: RangeSet::new(),
            context,
            allocation_pool,
        }
    }

    pub(super) fn access(
        &mut self,
        accesses: &[Access],
    ) {
        // TODO: more fine grained barriers
        if accesses.iter().any(|access| {
            ((access.flags.read || access.flags.write) && self.writes.overlaps(&access.range))
                || (access.flags.write && self.reads.overlaps(&access.range))
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
            if access.flags.read {
                self.reads.insert(access.range.clone());
            }
            if access.flags.write {
                self.writes.insert(access.range.clone());
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
    ) -> Result<Allocation<Metal>, MetalError> {
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
    ) -> Result<Allocation<Metal>, MetalError> {
        self.context.allocator.allocate(
            size,
            AllocationType::Pooled {
                pool: &self.allocation_pool,
                cpu_available: false,
            },
        )
    }

    fn encode_copy<Src: Buffer<Backend = Metal>, Dst: Buffer<Backend = Metal>>(
        &mut self,
        src: BufferRangeRef<Src>,
        dst: BufferRangeMut<Dst>,
    ) {
        let src_range = src.range();
        let dst_range = dst.range();
        assert_eq!(src_range.len(), dst_range.len());

        self.access(&[
            Access {
                range: src.buffer().gpu_address_subrange(src_range.clone()),
                flags: AccessFlags::read(),
            },
            Access {
                range: dst.buffer().gpu_address_subrange(dst_range.clone()),
                flags: AccessFlags::write(),
            },
        ]);

        self.compute_encoder.copy_from_buffer_source_offset_to_buffer_destination_offset_size(
            (src.buffer() as &dyn Buffer<Backend = Metal>).downcast(),
            src_range.start,
            (dst.buffer() as &dyn Buffer<Backend = Metal>).downcast(),
            dst_range.start,
            src_range.len(),
        );
    }

    fn encode_fill<Dst: Buffer<Backend = Metal>>(
        &mut self,
        dst: BufferRangeMut<Dst>,
        value: u8,
    ) {
        let range = dst.range();
        assert!(range.end > range.start);
        assert!(range.start.is_multiple_of(4) && range.end.is_multiple_of(4));

        self.access(&[Access {
            range: dst.buffer().gpu_address_subrange(range.clone()),
            flags: AccessFlags::write(),
        }]);

        self.compute_encoder.fill_buffer_range_value(
            (dst.buffer() as &dyn Buffer<Backend = Metal>).downcast(),
            range,
            value,
        );
    }

    // TODO: maybe port previous debug command_buffer labels
    fn push_debug_group(
        &mut self,
        name: &str,
    ) {
        let command_encoder: &ProtocolObject<dyn MTL4CommandEncoder> = self.compute_encoder.as_ref();
        command_encoder.push_debug_group(name);
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
    allocation_pool: Arc<AllocationPool<Metal>>,
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
    allocation_pool: Arc<AllocationPool<Metal>>,
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
    _allocation_pool: Arc<AllocationPool<Metal>>,
}

impl CommandBufferCompleted for MetalCommandBufferCompleted {
    type CommandBuffer = MetalCommandBuffer;

    fn gpu_execution_time(&self) -> Duration {
        self.gpu_execution_time
    }
}
