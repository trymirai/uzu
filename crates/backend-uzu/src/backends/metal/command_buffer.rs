use std::{cell::Cell, rc::Rc, time::Duration};

use metal::{
    MTLBlitCommandEncoder, MTLBlitCommandEncoderExt, MTLCommandBuffer, MTLCommandBufferExt, MTLCommandBufferHandler,
    MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLEvent,
};
use objc2::{Message, rc::Retained, runtime::ProtocolObject};

use super::Metal;
use crate::{
    backends::{
        common::{
            AccessFlags, Backend, BufferRangeMut, BufferRangeRef, CommandBuffer, CommandBufferCompleted,
            CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending,
        },
        metal::error::MetalError,
    },
    prelude::MetalContext,
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

fn command_buffer_result(command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), MetalError> {
    match (command_buffer.status(), command_buffer.error()) {
        (MTLCommandBufferStatus::Completed, None) => Ok(()),
        (status, Some(nserror)) => Err(MetalError::CommandBufferExecutionFailed(format!("{status:?}: {nserror:?}"))),
        (status, None) => Err(MetalError::CommandBufferExecutionFailed(format!("{status:?}"))),
    }
}

pub struct MetalCommandBufferInitial {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    context: Rc<MetalContext>,
}

impl MetalCommandBufferInitial {
    pub fn new(
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        context: Rc<MetalContext>,
    ) -> Self {
        Self {
            command_buffer,
            context,
        }
    }
}

impl CommandBufferInitial for MetalCommandBufferInitial {
    type CommandBuffer = MetalCommandBuffer;

    fn start_encoding(self) -> MetalCommandBufferEncoding {
        MetalCommandBufferEncoding {
            command_buffer: self.command_buffer,
            encoding_state: MetalCommandBufferEncodingEncodingState::None,
            context: self.context,
        }
    }
}

enum MetalCommandBufferEncodingEncodingState {
    None,
    Compute(Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>),
    Blit(Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>),
}

pub struct MetalCommandBufferEncoding {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoding_state: MetalCommandBufferEncodingEncodingState,
    context: Rc<MetalContext>,
}

impl MetalCommandBufferEncoding {
    fn ensure_none(&mut self) {
        match &mut self.encoding_state {
            MetalCommandBufferEncodingEncodingState::None => return,
            MetalCommandBufferEncodingEncodingState::Compute(compute_encoder) => compute_encoder.end_encoding(),
            MetalCommandBufferEncodingEncodingState::Blit(blit_encoder) => blit_encoder.end_encoding(),
        };

        self.encoding_state = MetalCommandBufferEncodingEncodingState::None;
    }

    pub(super) fn ensure_compute(&mut self) -> &mut Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> {
        if !matches!(self.encoding_state, MetalCommandBufferEncodingEncodingState::Compute(_)) {
            self.ensure_none();
            self.encoding_state = MetalCommandBufferEncodingEncodingState::Compute(
                self.command_buffer.compute_command_encoder().expect("Failed to create compute command encoder"),
            );
        }

        let MetalCommandBufferEncodingEncodingState::Compute(compute_encoder) = &mut self.encoding_state else {
            unreachable!()
        };
        compute_encoder
    }

    fn ensure_blit(&mut self) -> &mut Retained<ProtocolObject<dyn MTLBlitCommandEncoder>> {
        if !matches!(self.encoding_state, MetalCommandBufferEncodingEncodingState::Blit(_)) {
            self.ensure_none();
            self.encoding_state = MetalCommandBufferEncodingEncodingState::Blit(
                self.command_buffer.blit_command_encoder().expect("Failed to create blit command encoder"),
            );
        }

        let MetalCommandBufferEncodingEncodingState::Blit(blit_encoder) = &mut self.encoding_state else {
            unreachable!()
        };
        blit_encoder
    }
}

impl Drop for MetalCommandBufferEncoding {
    fn drop(&mut self) {
        self.ensure_none();
    }
}

impl CommandBufferEncoding for MetalCommandBufferEncoding {
    type CommandBuffer = MetalCommandBuffer;

    fn encode_copy(
        &mut self,
        src: BufferRangeRef<'_, <Metal as Backend>::DenseBuffer>,
        dst: BufferRangeMut<'_, <Metal as Backend>::DenseBuffer>,
    ) {
        let src_range = src.range();
        let dst_range = dst.range();
        let size = src_range.end - src_range.start;
        assert_eq!(size, dst_range.end - dst_range.start);

        self.ensure_blit().copy_buffer_to_buffer(src.buffer(), src_range.start, dst.buffer(), dst_range.start, size);
    }

    fn encode_fill(
        &mut self,
        dst: BufferRangeMut<'_, <Metal as Backend>::DenseBuffer>,
        value: u8,
    ) {
        let range = dst.range();
        assert!(range.end > range.start);
        assert!(range.start % 4 == 0 && range.end % 4 == 0);

        self.ensure_blit().fill_buffer_range_value(dst.buffer(), range, value);
    }

    fn encode_barrier(
        &mut self,
        _after: AccessFlags,
        _before: AccessFlags,
    ) {
    }

    fn encode_wait_for_event(
        &mut self,
        event: &Retained<ProtocolObject<dyn MTLEvent>>,
        value: u64,
    ) {
        self.ensure_none();
        self.command_buffer.encode_wait_for_event_value(event, value);
    }

    fn encode_signal_event(
        &mut self,
        event: &Retained<ProtocolObject<dyn MTLEvent>>,
        value: u64,
    ) {
        self.ensure_none();
        self.command_buffer.encode_signal_event_value(event, value);
    }

    fn add_completion_handler(
        &mut self,
        handler: impl FnOnce(Result<&MetalCommandBufferCompleted, MetalError>) + Send + 'static,
    ) {
        let cell = Cell::new(Some(handler));
        self.command_buffer.add_completed_handler(&MTLCommandBufferHandler::new(move |command_buffer| {
            let cbuf_ref = MetalCommandBufferCompleted {
                command_buffer: command_buffer.retain(),
            };
            cell.take().expect("completion handler called more than once")(
                command_buffer_result(command_buffer).map(|_| &cbuf_ref),
            )
        }));
    }

    fn end_encoding(mut self) -> <Self::CommandBuffer as CommandBuffer>::Executable {
        self.ensure_none();

        MetalCommandBufferExecutable {
            command_buffer: self.command_buffer.clone(),
            context: self.context.clone(),
        }
    }
}

pub struct MetalCommandBufferExecutable {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    context: Rc<MetalContext>,
}

impl CommandBufferExecutable for MetalCommandBufferExecutable {
    type CommandBuffer = MetalCommandBuffer;

    fn submit(self) -> MetalCommandBufferPending {
        let cmd_queue = self.command_buffer.command_queue();
        let wait_value = self.context.timeline_get_and_increment();

        {
            let wait_cmd_buffer = cmd_queue.command_buffer().expect("Failed to create command buffer");
            wait_cmd_buffer.encode_wait_for_event_value(self.context.timeline_event(), wait_value);
            wait_cmd_buffer.commit();
        }

        self.command_buffer.commit();

        {
            let wait_cmd_buffer = cmd_queue.command_buffer().expect("Failed to create command buffer");
            wait_cmd_buffer.encode_signal_event_value(self.context.timeline_event(), wait_value + 1);
            wait_cmd_buffer.commit();
        }

        MetalCommandBufferPending {
            command_buffer: self.command_buffer,
        }
    }
}

pub struct MetalCommandBufferPending {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

impl CommandBufferPending for MetalCommandBufferPending {
    type CommandBuffer = MetalCommandBuffer;

    fn wait_until_completed(self) -> Result<MetalCommandBufferCompleted, MetalError> {
        self.command_buffer.wait_until_completed();

        command_buffer_result(&self.command_buffer).map(|_| MetalCommandBufferCompleted {
            command_buffer: self.command_buffer,
        })
    }
}

pub struct MetalCommandBufferCompleted {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

impl CommandBufferCompleted for MetalCommandBufferCompleted {
    type CommandBuffer = MetalCommandBuffer;

    fn gpu_execution_time(&self) -> Duration {
        // They're always present, https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime?language=objc
        let start = self.command_buffer.gpu_start_time();
        let end = self.command_buffer.gpu_end_time();
        Duration::from_secs_f64(end - start)
    }
}
