use std::{ops::Range, rc::Rc, sync::mpsc, time::Duration};

use crate::backends::{
    common::{
        AccessFlags, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable,
        CommandBufferInitial, CommandBufferPending,
    },
    webgpu::{WebGPU, buffer::WebGPUBuffer, context::WebGPUContext, error::WebGPUError, event::WebGPUEvent},
};

pub struct WebGPUCommandBuffer;

impl CommandBuffer for WebGPUCommandBuffer {
    type Backend = WebGPU;

    type Initial = WebGPUCommandBufferInitial;
    type Encoding = WebGPUCommandBufferEncoding;
    type Executable = WebGPUCommandBufferExecutable;
    type Pending = WebGPUCommandBufferPending;
    type Completed = WebGPUCommandBufferCompleted;
}

pub struct WebGPUCommandBufferInitial {
    pub command_encoder: wgpu::CommandEncoder,
    pub context: Rc<WebGPUContext>,
}

impl CommandBufferInitial for WebGPUCommandBufferInitial {
    type CommandBuffer = WebGPUCommandBuffer;

    fn start_encoding(self) -> WebGPUCommandBufferEncoding {
        WebGPUCommandBufferEncoding {
            compute_pass: None,
            command_encoder: self.command_encoder,
            context: self.context,
        }
    }
}

pub struct WebGPUCommandBufferEncoding {
    compute_pass: Option<wgpu::ComputePass<'static>>,
    command_encoder: wgpu::CommandEncoder,
    context: Rc<WebGPUContext>,
}

impl WebGPUCommandBufferEncoding {
    fn ensure_none(&mut self) {
        self.compute_pass = None;
    }

    fn ensure_compute<'a>(&'a mut self) -> &'a mut wgpu::ComputePass<'static> {
        if self.compute_pass.is_none() {
            self.compute_pass =
                Some(self.command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default()).forget_lifetime())
        }

        self.compute_pass.as_mut().unwrap()
    }
}

impl CommandBufferEncoding for WebGPUCommandBufferEncoding {
    type CommandBuffer = WebGPUCommandBuffer;

    fn encode_copy(
        &mut self,
        src: &WebGPUBuffer,
        src_range: Range<usize>,
        dst: &mut WebGPUBuffer,
        dst_range: Range<usize>,
    ) {
        self.ensure_none();

        self.command_encoder.copy_buffer_to_buffer(
            &src.buffer,
            src_range.start as u64,
            &mut dst.buffer,
            dst_range.start as u64,
            (dst_range.end - dst_range.start) as u64,
        );
    }

    fn encode_fill(
        &mut self,
        dst: &mut WebGPUBuffer,
        range: Range<usize>,
        value: u8,
    ) {
        assert!(value == 0, "non-zero fill not supported");

        self.command_encoder.clear_buffer(&mut dst.buffer, range.start as u64, Some((range.end - range.start) as u64));
    }

    fn encode_barrier(
        &mut self,
        _after: AccessFlags,
        _before: AccessFlags,
    ) {
        // not needed on webgpu
    }

    fn encode_wait_for_event(
        &mut self,
        _event: &WebGPUEvent,
        _value: u64,
    ) {
        todo!("refactor required")
    }

    fn encode_signal_event(
        &mut self,
        _event: &WebGPUEvent,
        _value: u64,
    ) {
        todo!("refactor required")
    }

    fn add_completion_handler(
        &mut self,
        handler: impl FnOnce(Result<&WebGPUCommandBufferCompleted, WebGPUError>) + Send + 'static,
    ) {
        self.ensure_none();

        self.command_encoder.on_submitted_work_done(|| handler(Ok(&WebGPUCommandBufferCompleted {})));
    }

    fn end_encoding(mut self) -> WebGPUCommandBufferExecutable {
        self.ensure_none();

        WebGPUCommandBufferExecutable {
            command_buffer: self.command_encoder.finish(),
            context: self.context,
        }
    }
}

pub struct WebGPUCommandBufferExecutable {
    command_buffer: wgpu::CommandBuffer,
    context: Rc<WebGPUContext>,
}

impl CommandBufferExecutable for WebGPUCommandBufferExecutable {
    type CommandBuffer = WebGPUCommandBuffer;

    fn submit(self) -> WebGPUCommandBufferPending {
        let (sender, receiver) = mpsc::channel();

        self.command_buffer.on_submitted_work_done(move || {
            let _ = sender.send(());
        });
        self.context.queue.submit([self.command_buffer]);

        WebGPUCommandBufferPending {
            completed_signal: receiver,
        }
    }
}

pub struct WebGPUCommandBufferPending {
    completed_signal: mpsc::Receiver<()>,
}

impl CommandBufferPending for WebGPUCommandBufferPending {
    type CommandBuffer = WebGPUCommandBuffer;

    fn wait_until_completed(self) -> Result<WebGPUCommandBufferCompleted, WebGPUError> {
        self.completed_signal.recv().map_err(|_| WebGPUError::CommandBufferFailed)?;

        Ok(WebGPUCommandBufferCompleted {})
    }
}

pub struct WebGPUCommandBufferCompleted {}

impl CommandBufferCompleted for WebGPUCommandBufferCompleted {
    type CommandBuffer = WebGPUCommandBuffer;

    fn gpu_execution_time(&self) -> Duration {
        todo!()
    }
}
