use std::{
    sync::{Arc, mpsc},
    time::Duration,
};

use metal::{
    MTL4CommandAllocator, MTL4CommandBuffer, MTL4CommandEncoder, MTL4CommandEncoderExt, MTL4CommandQueueExt,
    MTL4CommitFeedback, MTL4CommitFeedbackExt, MTL4CommitFeedbackHandler, MTL4CommitOptions, MTL4ComputeCommandEncoder,
    MTL4ComputeCommandEncoderExt, MTL4VisibilityOptions, MTLStages,
};
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::backends::{
    common::{
        AccessFlags, Buffer, BufferRangeMut, BufferRangeRef, CommandBuffer, CommandBufferCompleted,
        CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending,
    },
    metal::{Metal, MetalContext, error::MetalError},
};

pub struct MetalCommandBuffer;

impl CommandBuffer for MetalCommandBuffer {
    type Backend = Metal;

    type Initial = MetalCommandBufferInitial;
    type Encoding = MetalCommandBufferEncoding;
    type Executable = MetalCommandBufferExecutable;
    type Pending = MetalCommandBufferPending;
    type Completed = MetalCommandBufferCompleted;
}

pub struct MetalCommandBufferInitial {
    command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    context: Arc<MetalContext>,
}

impl MetalCommandBufferInitial {
    pub fn new(
        command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
        command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
        context: Arc<MetalContext>,
    ) -> Self {
        Self {
            command_allocator,
            command_buffer,
            context,
        }
    }
}

impl CommandBufferInitial for MetalCommandBufferInitial {
    type CommandBuffer = MetalCommandBuffer;

    fn start_encoding(self) -> MetalCommandBufferEncoding {
        self.command_buffer.begin_command_buffer_with_allocator(&self.command_allocator);

        let compute_encoder = self.command_buffer.compute_command_encoder().unwrap();

        compute_encoder.barrier_after_queue_stages_before_stages_visibility_options(
            MTLStages::Dispatch | MTLStages::Blit | MTLStages::ResourceState,
            MTLStages::Dispatch | MTLStages::Blit,
            MTL4VisibilityOptions::Device,
        );

        MetalCommandBufferEncoding {
            command_allocator: self.command_allocator,
            command_buffer: self.command_buffer,
            compute_encoder,
            context: self.context,
        }
    }
}

pub struct MetalCommandBufferEncoding {
    command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    pub(super) compute_encoder: Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>,
    pub(super) context: Arc<MetalContext>,
}

impl From<AccessFlags> for MTLStages {
    fn from(val: AccessFlags) -> Self {
        let mut render_stages = MTLStages::empty();

        if val.compute_read || val.compute_write {
            render_stages |= MTLStages::Dispatch;
        }

        if val.copy_read || val.copy_write {
            render_stages |= MTLStages::Blit;
        }

        render_stages
    }
}

impl CommandBufferEncoding for MetalCommandBufferEncoding {
    type CommandBuffer = MetalCommandBuffer;

    fn encode_copy<Src: Buffer<Backend = Metal>, Dst: Buffer<Backend = Metal>>(
        &mut self,
        src: BufferRangeRef<Src>,
        dst: BufferRangeMut<Dst>,
    ) {
        let src_range = src.range();
        let dst_range = dst.range();
        assert_eq!(src_range.len(), dst_range.len());

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

        self.compute_encoder.fill_buffer_range_value(
            (dst.buffer() as &dyn Buffer<Backend = Metal>).downcast(),
            range,
            value,
        );
    }

    fn encode_barrier(
        &mut self,
        after: AccessFlags,
        before: AccessFlags,
    ) {
        self.compute_encoder.barrier_after_encoder_stages_before_encoder_stages_visibility_options(
            after.into(),
            before.into(),
            MTL4VisibilityOptions::Device,
        );
    }

    // TODO: maybe port previous debug encoder labels
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
            receiver,
        }
    }
}

pub struct MetalCommandBufferPending {
    _command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    _command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
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
        })
    }
}

pub struct MetalCommandBufferCompleted {
    gpu_execution_time: Duration,
}

impl CommandBufferCompleted for MetalCommandBufferCompleted {
    type CommandBuffer = MetalCommandBuffer;

    fn gpu_execution_time(&self) -> Duration {
        self.gpu_execution_time
    }
}
