//! The ANE programs (one per MLP shape or linear output width) and the adapters that run layers on
//! them: the `ane` crate decides what runs when, this executes its steps on the command buffer.

use std::{ffi::c_void, ptr, sync::Arc};

use ane::w8a8::{Event, Layer, Queue, QueueScope, Region, Step};
use metal::MTLSharedEvent;
use objc2::runtime::ProtocolObject;

use crate::{
    backends::{
        common::{
            Backend, CommandBuffer, CommandBufferEncoding,
            kernel::matmul::{ActivationFormat, MatmulA},
        },
        metal::{
            Metal,
            ane::{MIN_TOKENS, launch, region},
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            error::MetalError,
        },
    },
    encodable_block::{
        linear::{InputBasis, Linear, LinearEncodeError, LinearInput},
        mlp::{DenseMlp, Mlp, MlpEncodeError, MlpStarted},
    },
};

type Scratch = <Metal as Backend>::ScratchBuffer;

// SAFETY: `MetalCommandBufferEncoding` is the context's single compute queue adapter. Each
// operation is encoded in order and dispatch access declarations are forwarded to barrier tracking.
unsafe impl Queue for MetalCommandBufferEncoding {
    type Buffer = Scratch;
    type Error = MetalError;

    fn identity(&self) -> QueueScope {
        self.failure().ane_scope()
    }

    fn device(&self) -> *mut c_void {
        ptr::from_ref(&*self.context.device).cast_mut().cast()
    }

    fn region(
        &self,
        buffer: &Scratch,
    ) -> Result<Region, MetalError> {
        if !buffer.page().page().belongs_to(&self.context) {
            return Err(MetalError::CommandBufferExecution("buffer belongs to another Metal context".into()));
        }
        Ok(region(buffer))
    }

    fn allocate(
        &mut self,
        bytes: usize,
    ) -> Result<Scratch, MetalError> {
        self.allocate_scratch(bytes)
    }

    unsafe fn execute(
        &mut self,
        step: Step<'_>,
    ) -> Result<(), Self::Error> {
        match step {
            Step::Dispatch(dispatch) => launch(self, &dispatch),
            Step::Signal(event, value) => self.signal_event(shared_event(event), value),
            Step::Wait(event, value) => self.wait_event(shared_event(event), value),
        }
    }
}

fn shared_event(event: &Event) -> &ProtocolObject<dyn MTLSharedEvent> {
    // SAFETY: the pointer is an `id<MTLSharedEvent>`, live as long as `event`.
    unsafe { &*event.as_ptr().cast() }
}

/// A layer whose prefill batches run on the ANE; everything else (decode, small batches, a failed
/// ANE) runs on the wrapped GPU layer.
pub struct Offloaded<T> {
    pub inner: Arc<T>,
    pub layer: Layer,
}

fn execution_error(error: impl std::fmt::Display) -> MetalError {
    MetalError::CommandBufferExecution(error.to_string())
}

impl Mlp<Metal> for Offloaded<DenseMlp<Metal>> {
    fn encode(
        &self,
        input: Scratch,
        batch_dim: u32,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<Scratch, MlpEncodeError<Metal>> {
        self.encode_start(input, batch_dim, command_buffer)?.finish(command_buffer)
    }

    fn encode_start<'a>(
        &'a self,
        input: Scratch,
        batch_dim: u32,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<MlpStarted<'a, Metal>, MlpEncodeError<Metal>> {
        self.inner.encode_start(input, batch_dim, command_buffer)
    }

    fn input_basis(
        &self,
        batch_dim: u32,
    ) -> InputBasis {
        if batch_dim >= MIN_TOKENS && self.layer.accepts(batch_dim) {
            InputBasis::AneUnrotated
        } else {
            InputBasis::GpuExpected
        }
    }

    fn supports_unrotated_input(&self) -> bool {
        self.layer.accepts(MIN_TOKENS)
    }

    fn encode_start_for_basis<'a>(
        &'a self,
        input: Scratch,
        batch_dim: u32,
        basis: InputBasis,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<MlpStarted<'a, Metal>, MlpEncodeError<Metal>> {
        if basis == InputBasis::GpuExpected {
            return self.inner.encode_start(input, batch_dim, command_buffer);
        }
        if batch_dim < MIN_TOKENS || !self.layer.accepts(batch_dim) {
            return Err(MlpEncodeError::InputBasisUnavailable);
        }
        let command_buffer = &mut *command_buffer.debug_group("mlp (ane)");
        let call = self
            .layer
            .start(&input, batch_dim, command_buffer)
            .map_err(execution_error)
            .map_err(MlpEncodeError::Backend)?;
        Ok(MlpStarted::Handoff(Box::new(move |command_buffer: &mut MetalCommandBufferEncoding| {
            let command_buffer = &mut *command_buffer.debug_group("mlp (ane)");
            call.finish(command_buffer).map_err(execution_error).map_err(MlpEncodeError::Backend)
        })))
    }
}

impl Linear<Metal> for Offloaded<Box<dyn Linear<Metal>>> {
    fn encode_input(
        &self,
        input: LinearInput<Metal>,
        batch_dim: u32,
        command_buffer: &mut <<Metal as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Scratch, LinearEncodeError<Metal>> {
        if let MatmulA::FullPrecision {
            values,
            offset: 0,
        } = &input
            && batch_dim >= MIN_TOKENS
            && self.layer.accepts(batch_dim)
        {
            let command_buffer = &mut *command_buffer.debug_group("linear (ane)");
            return self
                .layer
                .start(values, batch_dim, command_buffer)
                .and_then(|call| call.finish(command_buffer))
                .map_err(execution_error)
                .map_err(LinearEncodeError::Backend);
        }
        self.inner.encode_input(input, batch_dim, command_buffer)
    }

    fn encode_for_basis(
        &self,
        input: Scratch,
        batch_dim: u32,
        basis: InputBasis,
        command_buffer: &mut <<Metal as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Scratch, LinearEncodeError<Metal>> {
        if basis == InputBasis::GpuExpected {
            return self.inner.encode(input, batch_dim, command_buffer);
        }
        if batch_dim < MIN_TOKENS || !self.layer.accepts(batch_dim) {
            return Err(LinearEncodeError::InputBasisUnavailable);
        }
        let command_buffer = &mut *command_buffer.debug_group("linear (ane)");
        self.layer
            .start(&input, batch_dim, command_buffer)
            .and_then(|call| call.finish(command_buffer))
            .map_err(execution_error)
            .map_err(LinearEncodeError::Backend)
    }

    fn select_activation_format(
        &self,
        batch_dim: u32,
        output_dim: Option<u32>,
        context: &MetalContext,
    ) -> ActivationFormat {
        self.inner.select_activation_format(batch_dim, output_dim, context)
    }

    fn input_basis(
        &self,
        batch_dim: u32,
    ) -> InputBasis {
        if batch_dim >= MIN_TOKENS && self.layer.accepts(batch_dim) {
            InputBasis::AneUnrotated
        } else {
            InputBasis::GpuExpected
        }
    }

    fn supports_unrotated_input(&self) -> bool {
        self.layer.accepts(MIN_TOKENS)
    }
}
