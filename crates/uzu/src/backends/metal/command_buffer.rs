use metal::{
    MTLBlitCommandEncoder, MTLBlitCommandEncoderExt, MTLBuffer, MTLCommandBuffer, MTLCommandBufferExt,
    MTLCommandBufferHandler, MTLCommandBufferStatus, MTLCommandEncoder, MTLComputeCommandEncoder, MTLEvent,
};
use objc2::{Message, rc::Retained, runtime::ProtocolObject};
use std::cell::Cell;

use super::Metal;
use crate::backends::{
    common::{
        CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial,
        CommandBufferPending,
    },
    metal::error::MetalError,
};

enum MetalCommandBufferEncodingState {
    None,
    Compute(Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>),
    Blit(Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>),
}

pub struct MetalCommandBuffer {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoding_state: MetalCommandBufferEncodingState,
}

pub type MetalCommandBufferInitial = MetalCommandBuffer;
pub type MetalCommandBufferEncoding = MetalCommandBuffer;
pub type MetalCommandBufferExecutable = MetalCommandBuffer;
pub type MetalCommandBufferPending = MetalCommandBuffer;
pub type MetalCommandBufferCompleted = MetalCommandBuffer;

impl Clone for MetalCommandBuffer {
    fn clone(&self) -> Self {
        Self {
            command_buffer: self.command_buffer.clone(),
            encoding_state: MetalCommandBufferEncodingState::None,
        }
    }
}

impl CommandBuffer for MetalCommandBuffer {
    type Backend = Metal;

    type Initial = MetalCommandBuffer;
    type Encoding = MetalCommandBuffer;
    type Executable = MetalCommandBuffer;
    type Pending = MetalCommandBuffer;
    type Completed = MetalCommandBuffer;
}

fn command_buffer_result(command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), MetalError> {
    match (command_buffer.status(), command_buffer.error()) {
        (MTLCommandBufferStatus::Completed, None) => Ok(()),
        (status, Some(nserror)) => Err(MetalError::CommandBufferExecutionFailed(format!("{status:?}: {nserror:?}"))),
        (status, None) => Err(MetalError::CommandBufferExecutionFailed(format!("{status:?}"))),
    }
}

impl MetalCommandBuffer {
    pub fn new(command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Self {
        Self {
            command_buffer,
            encoding_state: MetalCommandBufferEncodingState::None,
        }
    }

    fn ensure_none(&mut self) {
        match &mut self.encoding_state {
            MetalCommandBufferEncodingState::None => return,
            MetalCommandBufferEncodingState::Compute(compute_encoder) => compute_encoder.end_encoding(),
            MetalCommandBufferEncodingState::Blit(blit_encoder) => blit_encoder.end_encoding(),
        };

        self.encoding_state = MetalCommandBufferEncodingState::None;
    }

    pub(crate) fn ensure_compute(&mut self) -> &mut Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> {
        if !matches!(self.encoding_state, MetalCommandBufferEncodingState::Compute(_)) {
            self.ensure_none();
            self.encoding_state = MetalCommandBufferEncodingState::Compute(
                self.command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder"),
            );
        }

        let MetalCommandBufferEncodingState::Compute(compute_encoder) = &mut self.encoding_state else {
            unreachable!()
        };
        compute_encoder
    }

    pub(crate) fn ensure_blit(&mut self) -> &mut Retained<ProtocolObject<dyn MTLBlitCommandEncoder>> {
        if !matches!(self.encoding_state, MetalCommandBufferEncodingState::Blit(_)) {
            self.ensure_none();
            self.encoding_state = MetalCommandBufferEncodingState::Blit(
                self.command_buffer.new_blit_command_encoder().expect("Failed to create blit command encoder"),
            );
        }

        let MetalCommandBufferEncodingState::Blit(blit_encoder) = &mut self.encoding_state else {
            unreachable!()
        };
        blit_encoder
    }

    pub fn with_compute_encoder<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        self.ensure_compute();
        f(self)
    }

    pub fn with_copy_encoder<T>(
        &mut self,
        f: impl FnOnce(&mut Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>) -> T,
    ) -> T {
        f(self.ensure_blit())
    }

    pub fn submit(&mut self) {
        self.ensure_none();
        self.command_buffer.commit();
    }

    pub fn wait_until_completed(&self) -> Result<(), MetalError> {
        self.command_buffer.wait_until_completed();
        command_buffer_result(&self.command_buffer)?;
        Ok(())
    }

    pub fn is_completed(&self) -> bool {
        matches!(self.command_buffer.status(), MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error)
    }

    pub fn gpu_execution_time_ms(&self) -> Option<f64> {
        match (self.command_buffer.gpu_start_time(), self.command_buffer.gpu_end_time()) {
            (Some(start), Some(end)) => Some((end - start) * 1000.0),
            _ => None,
        }
    }
}

impl Drop for MetalCommandBuffer {
    fn drop(&mut self) {
        self.ensure_none();
    }
}

impl CommandBufferInitial for MetalCommandBuffer {
    type CommandBuffer = MetalCommandBuffer;

    fn start_encoding(self) -> MetalCommandBuffer {
        self
    }
}

impl CommandBufferEncoding for MetalCommandBuffer {
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
        handler: impl FnOnce(Result<&MetalCommandBuffer, MetalError>) + 'static,
    ) {
        let cell = Cell::new(Some(handler));
        self.command_buffer.add_completed_handler(&MTLCommandBufferHandler::new(move |command_buffer| {
            let completed = MetalCommandBuffer::new(command_buffer.retain());
            cell.take().expect("completion handler called more than once")(
                command_buffer_result(command_buffer).map(|_| &completed),
            );
        }));
    }

    fn end_encoding(mut self) -> MetalCommandBuffer {
        self.ensure_none();
        self
    }
}

impl CommandBufferExecutable for MetalCommandBuffer {
    type CommandBuffer = MetalCommandBuffer;

    fn submit(mut self) -> MetalCommandBuffer {
        Self::submit(&mut self);
        self
    }
}

impl CommandBufferPending for MetalCommandBuffer {
    type CommandBuffer = MetalCommandBuffer;

    fn wait_until_completed(self) -> Result<MetalCommandBuffer, MetalError> {
        Self::wait_until_completed(&self)?;
        Ok(self)
    }
}

impl CommandBufferCompleted for MetalCommandBuffer {
    type CommandBuffer = MetalCommandBuffer;

    fn is_completed(&self) -> bool {
        Self::is_completed(self)
    }

    fn gpu_execution_time_ms(&self) -> Option<f64> {
        Self::gpu_execution_time_ms(self)
    }
}
