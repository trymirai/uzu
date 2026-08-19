use std::{
    iter::{chain, once},
    sync::{Arc, LazyLock},
    time::Duration,
};

use itertools::Itertools;
use metal::{
    MTLBlitCommandEncoder, MTLBlitCommandEncoderExt, MTLCommandBuffer, MTLCommandBufferExt, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandEncoderExt, MTLComputeCommandEncoder,
};
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::backends::{
    common::{
        AccessFlags, Buffer, BufferRangeMut, BufferRangeRef, CommandBuffer, CommandBufferCompleted,
        CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending,
    },
    metal::{Metal, MetalContext, error::MetalError},
};

static DEBUG_ENCODER_LABELS: LazyLock<bool> = LazyLock::new(|| std::env::var("UZU_METAL_DEBUG_ENCODER_LABELS").is_ok());

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
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    context: Arc<MetalContext>,
}

impl MetalCommandBufferInitial {
    pub fn new(
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        context: Arc<MetalContext>,
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
            debug_group_stack: vec![],
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
    debug_group_stack: Vec<String>,
    context: Arc<MetalContext>,
}

impl MetalCommandBufferEncoding {
    fn ensure_none(&mut self) {
        let encoder: &ProtocolObject<dyn MTLCommandEncoder> = match &self.encoding_state {
            MetalCommandBufferEncodingEncodingState::None => return,
            MetalCommandBufferEncodingEncodingState::Compute(compute_encoder) => compute_encoder.as_ref(),
            MetalCommandBufferEncodingEncodingState::Blit(blit_encoder) => blit_encoder.as_ref(),
        };

        for _ in &self.debug_group_stack {
            encoder.pop_debug_group();
        }

        encoder.end_encoding();

        self.encoding_state = MetalCommandBufferEncodingEncodingState::None;
    }

    pub(super) fn ensure_compute(&mut self) -> &mut Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> {
        if !matches!(self.encoding_state, MetalCommandBufferEncodingEncodingState::Compute(_)) {
            self.ensure_none();
            let compute_encoder =
                self.command_buffer.compute_command_encoder().expect("Failed to create compute command encoder");
            self.ensure_common(compute_encoder.as_ref());
            self.encoding_state = MetalCommandBufferEncodingEncodingState::Compute(compute_encoder);
        }

        let MetalCommandBufferEncodingEncodingState::Compute(compute_encoder) = &mut self.encoding_state else {
            unreachable!()
        };
        compute_encoder
    }

    fn ensure_blit(&mut self) -> &mut Retained<ProtocolObject<dyn MTLBlitCommandEncoder>> {
        if !matches!(self.encoding_state, MetalCommandBufferEncodingEncodingState::Blit(_)) {
            self.ensure_none();
            let blit_encoder =
                self.command_buffer.blit_command_encoder().expect("Failed to create blit command encoder");
            self.ensure_common(blit_encoder.as_ref());
            self.encoding_state = MetalCommandBufferEncodingEncodingState::Blit(blit_encoder);
        }

        let MetalCommandBufferEncodingEncodingState::Blit(blit_encoder) = &mut self.encoding_state else {
            unreachable!()
        };
        blit_encoder
    }

    fn ensure_common(
        &self,
        encoder: &ProtocolObject<dyn MTLCommandEncoder>,
    ) {
        let command_buffer_label = self.command_buffer.label();
        let label = if *DEBUG_ENCODER_LABELS && (command_buffer_label.is_some() || !self.debug_group_stack.is_empty()) {
            Some(
                chain(
                    once(command_buffer_label.as_deref()),
                    self.debug_group_stack.iter().map(|label| Some(label.as_str())),
                )
                .flatten()
                .join("."),
            )
        } else {
            command_buffer_label
        };
        if label.is_some() {
            encoder.set_label(label.as_deref());
        }
        for debug_group in &self.debug_group_stack {
            encoder.push_debug_group(debug_group);
        }
    }
}

impl Drop for MetalCommandBufferEncoding {
    fn drop(&mut self) {
        self.ensure_none();
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

        self.ensure_blit().copy_buffer_to_buffer(
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

        self.ensure_blit().fill_buffer_range_value(
            (dst.buffer() as &dyn Buffer<Backend = Metal>).downcast(),
            range,
            value,
        );
    }

    fn encode_barrier(
        &mut self,
        _after: AccessFlags,
        _before: AccessFlags,
    ) {
    }

    fn push_debug_group(
        &mut self,
        name: &str,
    ) {
        if !*DEBUG_ENCODER_LABELS {
            return;
        }

        self.ensure_none();

        self.debug_group_stack.push(name.to_string());

        match &self.encoding_state {
            MetalCommandBufferEncodingEncodingState::None => (),
            MetalCommandBufferEncodingEncodingState::Compute(compute_encoder) => compute_encoder.push_debug_group(name),
            MetalCommandBufferEncodingEncodingState::Blit(blit_encoder) => {
                let encoder: &ProtocolObject<dyn MTLCommandEncoder> = blit_encoder.as_ref();
                encoder.push_debug_group(name);
            },
        }
    }

    fn pop_debug_group(&mut self) {
        if !*DEBUG_ENCODER_LABELS {
            return;
        }

        self.ensure_none();

        self.debug_group_stack.pop().expect("debug group stack underflow");

        match &self.encoding_state {
            MetalCommandBufferEncodingEncodingState::None => (),
            MetalCommandBufferEncodingEncodingState::Compute(compute_encoder) => compute_encoder.pop_debug_group(),
            MetalCommandBufferEncodingEncodingState::Blit(blit_encoder) => {
                let encoder: &ProtocolObject<dyn MTLCommandEncoder> = blit_encoder.as_ref();
                encoder.pop_debug_group();
            },
        }
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
    context: Arc<MetalContext>,
}

impl CommandBufferExecutable for MetalCommandBufferExecutable {
    type CommandBuffer = MetalCommandBuffer;

    fn submit(self) -> MetalCommandBufferPending {
        self.context.submit_compute(&self.command_buffer);

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

        match (self.command_buffer.status(), self.command_buffer.error()) {
            (MTLCommandBufferStatus::Completed, None) => (),
            (status, Some(nserror)) => {
                return Err(MetalError::CommandBufferExecutionFailed(format!("{status:?}: {nserror:?}")));
            },
            (status, None) => return Err(MetalError::CommandBufferExecutionFailed(format!("{status:?}"))),
        }

        Ok(MetalCommandBufferCompleted {
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
