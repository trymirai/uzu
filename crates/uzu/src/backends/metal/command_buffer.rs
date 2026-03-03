use std::{cell::Cell, ops::Deref};

use metal::{
    MTLCommandBuffer, MTLCommandBufferExt, MTLCommandBufferHandler, MTLCommandBufferStatus, MTLCommandEncoder, MTLEvent,
};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::Metal;
use crate::backends::{common::CommandBuffer, metal::error::MetalError};

fn command_buffer_result(command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), MetalError> {
    match (command_buffer.status(), command_buffer.error()) {
        (MTLCommandBufferStatus::Completed, None) => Ok(()),
        (status, Some(nserror)) => Err(MetalError::CommandBufferExecutionFailed(format!("{status:?}: {nserror:?}"))),
        (status, None) => Err(MetalError::CommandBufferExecutionFailed(format!("{status:?}"))),
    }
}

impl CommandBuffer for Retained<ProtocolObject<dyn MTLCommandBuffer>> {
    type Backend = Metal;

    fn push_debug_group(&mut self, label: &str) {
        MTLCommandBufferExt::push_debug_group(&**self, label);
    }

    fn pop_debug_group(&mut self) {
        MTLCommandBufferExt::pop_debug_group(&**self);
    }

    fn with_compute_encoder<T>(
        &mut self,
        callback: impl FnOnce(&mut <Self::Backend as crate::backends::common::Backend>::ComputeEncoder) -> T,
    ) -> T {
        let mut encoder = self.new_compute_command_encoder().expect("Failed to create compute command encoder");

        let ret = callback(&mut encoder);

        encoder.end_encoding();

        ret
    }

    fn with_copy_encoder<T>(
        &mut self,
        callback: impl FnOnce(&mut <Self::Backend as crate::backends::common::Backend>::CopyEncoder) -> T,
    ) -> T {
        let mut encoder = self.new_blit_command_encoder().expect("Failed to create blit command encoder");

        let ret = callback(&mut encoder);

        encoder.end_encoding();

        ret
    }

    fn encode_wait_for_event(
        &mut self,
        event: &Retained<ProtocolObject<dyn MTLEvent>>,
        value: u64,
    ) {
        self.encode_wait_for_event_value(event, value);
    }

    fn encode_signal_event(
        &mut self,
        event: &Retained<ProtocolObject<dyn MTLEvent>>,
        value: u64,
    ) {
        self.encode_signal_event_value(event, value);
    }

    fn add_completion_handler(
        &mut self,
        handler: impl FnOnce(Result<(), MetalError>) + 'static,
    ) {
        let cell = Cell::new(Some(handler));
        self.deref().add_completed_handler(&MTLCommandBufferHandler::new(move |command_buffer| {
            cell.take().expect("completion handler called more than once")(command_buffer_result(command_buffer))
        }));
    }

    fn submit(&mut self) {
        self.commit();
    }

    fn wait_until_completed(&self) -> Result<(), MetalError> {
        self.deref().wait_until_completed();
        command_buffer_result(self)
    }

    fn gpu_execution_time_ms(&self) -> Option<f64> {
        match (self.kernel_start_time(), self.kernel_end_time()) {
            (Some(start), Some(end)) => Some((end - start) * 1000.0),
            _ => None,
        }
    }
}
