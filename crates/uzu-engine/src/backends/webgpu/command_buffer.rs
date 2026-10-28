#[cfg(not(target_arch = "wasm32"))]
use std::thread;
use std::{sync::Arc, time::Duration};

use futures::channel::oneshot;

use crate::backends::{
    common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding,
        CommandBufferExecutable, CommandBufferPending, TimestampSpan, allocator::bump::BumpAllocator,
    },
    webgpu::{
        WebGPU,
        buffer::{
            binding_parts,
            dense::{WebGPUDenseBufferCpuGpu, WebGPUDenseBufferGpu},
        },
        context::WebGPUContext,
        error::WebGPUError,
    },
};

pub struct WebGPUCommandBuffer;

impl CommandBuffer for WebGPUCommandBuffer {
    type Backend = WebGPU;

    type Encoding = WebGPUCommandBufferEncoding;
    type Executable = WebGPUCommandBufferExecutable;
    type Pending = WebGPUCommandBufferPending;
    type Completed = WebGPUCommandBufferCompleted;
}

pub struct WebGPUCommandBufferEncoding {
    compute_pass: Option<wgpu::ComputePass<'static>>,
    command_encoder: wgpu::CommandEncoder,
    constant_allocator: BumpAllocator<WebGPUDenseBufferCpuGpu>,
    allocation_pool: Arc<<WebGPU as Backend>::AllocationPool>,
    context: Arc<WebGPUContext>,
}

// TODO: workaround for wgpu trait solver overflow
unsafe impl Send for WebGPUCommandBufferEncoding {}

impl WebGPUCommandBufferEncoding {
    pub fn new(
        context: Arc<WebGPUContext>,
        name: Option<&str>,
        allocation_pool: Arc<<WebGPU as Backend>::AllocationPool>,
    ) -> Self {
        let command_encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: name,
        });

        let constant_allocator = BumpAllocator::new(256 * 1024, wgpu::COPY_BUFFER_ALIGNMENT as usize);

        Self {
            compute_pass: None,
            command_encoder,
            constant_allocator,
            allocation_pool,
            context,
        }
    }

    fn ensure_none(&mut self) {
        self.compute_pass = None;
    }

    pub(super) fn ensure_compute(&mut self) -> &mut wgpu::ComputePass<'static> {
        if self.compute_pass.is_none() {
            let compute_pass = self.command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            self.compute_pass = Some(compute_pass.forget_lifetime());
        }

        self.compute_pass.as_mut().unwrap()
    }
}

impl CommandBufferEncoding for WebGPUCommandBufferEncoding {
    type CommandBuffer = WebGPUCommandBuffer;

    fn context(&self) -> &WebGPUContext {
        &self.context
    }

    fn allocate_constant(
        &mut self,
        size: usize,
    ) -> Result<<WebGPU as Backend>::ConstantBuffer, WebGPUError> {
        let alignment = usize::min(size, 64)
            .next_power_of_two()
            .max(self.context.limits.min_uniform_buffer_offset_alignment as usize)
            .max(self.context.limits.min_storage_buffer_offset_alignment as usize)
            .max(wgpu::COPY_BUFFER_ALIGNMENT as usize);
        self.constant_allocator.allocate(size, alignment, |size| WebGPUDenseBufferCpuGpu::new(&self.context, size))
    }

    fn allocate_scratch(
        &mut self,
        size: usize,
    ) -> Result<<WebGPU as Backend>::ScratchBuffer, WebGPUError> {
        self.allocation_pool.allocate(size, |size| WebGPUDenseBufferGpu::new(&self.context, size))
    }

    fn encode_copy(
        &mut self,
        src: impl BufferRef<Backend = WebGPU>,
        dst: impl BufferMut<Backend = WebGPU>,
    ) {
        self.ensure_none();
        let (src, src_range) = src.parts();
        let (dst, dst_range) = dst.parts();
        let (src, src_offset, src_size) = binding_parts(src, src_range);
        let (dst, dst_offset, dst_size) = binding_parts(dst, dst_range);
        assert_eq!(src_size, dst_size);
        self.command_encoder.copy_buffer_to_buffer(src, src_offset as u64, dst, dst_offset as u64, dst_size as u64);
    }

    fn encode_fill(
        &mut self,
        dst: impl BufferMut<Backend = WebGPU>,
        value: u8,
    ) {
        assert!(value == 0, "non-zero fill not supported yet"); // TODO
        self.ensure_none();
        let (dst, range) = dst.parts();
        let (dst, offset, size) = binding_parts(dst, range);
        self.command_encoder.clear_buffer(dst, offset as u64, Some(size as u64));
    }

    fn debug_group(
        &mut self,
        _name: &str,
    ) -> <WebGPU as Backend>::DebugGroupSpan<'_> {
        self
    }

    fn debug_point(
        &mut self,
        _name: &str,
    ) {
    }

    fn with_timestamps(self) -> Self {
        unimplemented!()
    }

    fn end_encoding(mut self) -> WebGPUCommandBufferExecutable {
        self.ensure_none();

        assert!(self.constant_allocator.is_done(), "attempted to end encoding while constants are still alive");

        WebGPUCommandBufferExecutable {
            command_buffer: self.command_encoder.finish(),
            context: self.context,
        }
    }
}

pub struct WebGPUCommandBufferExecutable {
    command_buffer: wgpu::CommandBuffer,
    context: Arc<WebGPUContext>,
}

impl CommandBufferExecutable for WebGPUCommandBufferExecutable {
    type CommandBuffer = WebGPUCommandBuffer;

    fn submit(self) -> WebGPUCommandBufferPending {
        let (sender, receiver) = oneshot::channel();

        self.command_buffer.on_submitted_work_done(move || {
            let _ = sender.send(());
        });

        let submission_index = self.context.queue.submit([self.command_buffer]);

        #[cfg(not(target_arch = "wasm32"))]
        thread::spawn(move || {
            let _ = self.context.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: None,
            });
        });

        #[cfg(target_arch = "wasm32")]
        let _ = submission_index;

        WebGPUCommandBufferPending {
            completed_signal: receiver,
        }
    }
}

pub struct WebGPUCommandBufferPending {
    completed_signal: oneshot::Receiver<()>,
}

impl CommandBufferPending for WebGPUCommandBufferPending {
    type CommandBuffer = WebGPUCommandBuffer;

    async fn wait_until_completed(self) -> Result<WebGPUCommandBufferCompleted, WebGPUError> {
        self.completed_signal.await.map_err(|_| WebGPUError::CommandBuffer)?;

        Ok(WebGPUCommandBufferCompleted)
    }
}

pub struct WebGPUCommandBufferCompleted;

impl CommandBufferCompleted for WebGPUCommandBufferCompleted {
    type CommandBuffer = WebGPUCommandBuffer;

    fn gpu_execution_time(&self) -> Option<Duration> {
        None
    }

    fn timestamps(&self) -> &[TimestampSpan] {
        unimplemented!()
    }
}
