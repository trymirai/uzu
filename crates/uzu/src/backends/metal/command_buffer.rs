use std::{cell::Cell, time::Duration};

use metal::{
    MTLBlitCommandEncoder, MTLBlitCommandEncoderExt, MTLBuffer, MTLCommandBuffer, MTLCommandBufferExt,
    MTLCommandBufferHandler, MTLCommandBufferStatus, MTLCommandEncoder, MTLComputeCommandEncoder, MTLEvent,
};
use objc2::{Message, rc::Retained, runtime::ProtocolObject};

use super::Metal;
use crate::backends::{
    common::{
        CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial,
        CommandBufferPending,
    },
    metal::error::MetalError,
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
}

impl MetalCommandBufferInitial {
    pub fn new(command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Self {
        Self {
            command_buffer,
        }
    }
}

impl CommandBufferInitial for MetalCommandBufferInitial {
    type CommandBuffer = MetalCommandBuffer;

    fn start_encoding(self) -> MetalCommandBufferEncoding {
        MetalCommandBufferEncoding {
            command_buffer: self.command_buffer,
            encoding_state: MetalCommandBufferEncodingEncodingState::None,
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
                self.command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder"),
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
                self.command_buffer.new_blit_command_encoder().expect("Failed to create blit command encoder"),
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
        src: &Retained<ProtocolObject<dyn MTLBuffer>>,
        dst: &mut Retained<ProtocolObject<dyn MTLBuffer>>,
        size: usize,
    ) {
        assert!(src.length() >= size && dst.length() >= size);

        self.ensure_blit().copy_buffer_to_buffer(src, 0, dst, 0, size);
    }

    fn encode_fill(
        &mut self,
        dst: &mut Retained<ProtocolObject<dyn MTLBuffer>>,
        range: std::ops::Range<usize>,
        value: u8,
    ) {
        assert!(range.end > range.start && range.end <= dst.length());
        assert!(range.start % 4 == 0 && range.end % 4 == 0);

        self.ensure_blit().fill_buffer_range_value(dst, range, value);
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
        handler: impl FnOnce(Result<&MetalCommandBufferCompleted, MetalError>) + 'static,
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
        }
    }
}

pub struct MetalCommandBufferExecutable {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

impl CommandBufferExecutable for MetalCommandBufferExecutable {
    type CommandBuffer = MetalCommandBuffer;

    fn submit(self) -> MetalCommandBufferPending {
        self.command_buffer.commit();

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

    fn gpu_execution_time(&self) -> Option<Duration> {
        match (self.command_buffer.gpu_start_time(), self.command_buffer.gpu_end_time()) {
            (Some(start), Some(end)) => Some(Duration::from_secs_f64(end - start)),
            _ => None,
        }
    }

}
