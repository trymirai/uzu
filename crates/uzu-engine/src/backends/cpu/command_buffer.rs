use std::{
    sync::{Arc, mpsc},
    time::{Duration, Instant},
};

use crate::{
    backends::{
        common::{
            Backend, BufferCpuAccessible, BufferMut, BufferRef, CommandBuffer, CommandBufferCompleted,
            CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, allocator::AllocationType,
        },
        cpu::{Cpu, buffer::CpuBufferExt, context::CpuContext, error::CpuError},
    },
    utils::pointers::{SendPtr, SendPtrMut},
};

pub struct CpuCommandBuffer;

impl CommandBuffer for CpuCommandBuffer {
    type Backend = Cpu;

    type Encoding = CpuCommandBufferEncoding;
    type Executable = CpuCommandBufferExecutable;
    type Pending = CpuCommandBufferPending;
    type Completed = CpuCommandBufferCompleted;
}

pub struct CpuCommandBufferEncoding {
    commands: Vec<Box<dyn FnOnce() + Send>>,
    context: Arc<CpuContext>,
    allocation_pool: Arc<<Cpu as Backend>::AllocationPool>,
}

impl CpuCommandBufferEncoding {
    pub fn new(
        context: Arc<CpuContext>,
        allocation_pool: Arc<<Cpu as Backend>::AllocationPool>,
    ) -> CpuCommandBufferEncoding {
        CpuCommandBufferEncoding {
            commands: Vec::new(),
            context,
            allocation_pool,
        }
    }

    pub fn push_command(
        &mut self,
        command: impl FnOnce() + Send + 'static,
    ) {
        self.commands.push(Box::new(command))
    }
}

impl CommandBufferEncoding for CpuCommandBufferEncoding {
    type CommandBuffer = CpuCommandBuffer;

    fn context(&self) -> &CpuContext {
        &self.context
    }

    fn allocate_constant(
        &mut self,
        size: usize,
    ) -> Result<<Cpu as Backend>::ConstantBuffer, CpuError> {
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
    ) -> Result<<Cpu as Backend>::ScratchBuffer, CpuError> {
        self.context.allocator.allocate(
            size,
            AllocationType::Pooled {
                pool: &self.allocation_pool,
                cpu_available: false,
            },
        )
    }

    fn encode_copy(
        &mut self,
        src: impl BufferRef<Backend = Cpu>,
        dst: impl BufferMut<Backend = Cpu>,
    ) {
        let (src, src_range) = src.parts();
        let (dst, dst_range) = dst.parts();
        assert_eq!(src_range.iter().len(), dst_range.iter().len());

        let src_buffer = src.downcast();
        let src_ptr = SendPtr(unsafe { src_buffer.cpu_ptr().as_ptr().cast::<u8>().add(src_range.start) });

        let dst_buffer = dst.downcast();
        let dst_ptr = SendPtrMut(unsafe { dst_buffer.cpu_ptr().as_ptr().cast::<u8>().add(dst_range.start) });

        self.push_command(move || unsafe {
            std::ptr::copy(src_ptr.as_ptr(), dst_ptr.as_ptr(), src_range.iter().len());
        });
    }

    fn encode_fill(
        &mut self,
        dst: impl BufferMut<Backend = Cpu>,
        value: u8,
    ) {
        let (dst, range) = dst.parts();
        let size = range.iter().len();
        let dst = SendPtrMut(unsafe { dst.downcast().cpu_ptr().as_ptr().cast::<u8>().add(range.start) });
        self.push_command(move || unsafe {
            dst.as_ptr().write_bytes(value, size);
        });
    }

    fn push_debug_group(
        &mut self,
        _name: &str,
    ) {
    }

    fn pop_debug_group(&mut self) {}

    fn end_encoding(self) -> CpuCommandBufferExecutable {
        CpuCommandBufferExecutable {
            commands: self.commands,
            context: self.context,
            allocation_pool: self.allocation_pool,
        }
    }
}

pub struct CpuCommandBufferExecutable {
    commands: Vec<Box<dyn FnOnce() + Send>>,
    context: Arc<CpuContext>,
    allocation_pool: Arc<<Cpu as Backend>::AllocationPool>,
}

impl CommandBufferExecutable for CpuCommandBufferExecutable {
    type CommandBuffer = CpuCommandBuffer;

    fn submit(self) -> CpuCommandBufferPending {
        let (return_sender, return_receiver) = mpsc::channel();

        let allocation_pool = self.allocation_pool;
        self.context
            .command_queue
            .send(Box::new(move || {
                let start = Instant::now();

                for command in self.commands {
                    command()
                }

                let gpu_execution_time = start.elapsed();

                let completed = CpuCommandBufferCompleted {
                    gpu_execution_time,
                    _allocation_pool: allocation_pool,
                };

                let _ = return_sender.send(completed);
            }))
            .unwrap();

        CpuCommandBufferPending {
            return_receiver,
        }
    }
}

pub struct CpuCommandBufferPending {
    return_receiver: mpsc::Receiver<CpuCommandBufferCompleted>,
}

impl CommandBufferPending for CpuCommandBufferPending {
    type CommandBuffer = CpuCommandBuffer;

    fn wait_until_completed(self) -> Result<CpuCommandBufferCompleted, CpuError> {
        Ok(self.return_receiver.recv()?)
    }
}

pub struct CpuCommandBufferCompleted {
    gpu_execution_time: Duration,
    _allocation_pool: Arc<<Cpu as Backend>::AllocationPool>,
}

impl CommandBufferCompleted for CpuCommandBufferCompleted {
    type CommandBuffer = CpuCommandBuffer;

    fn gpu_execution_time(&self) -> Duration {
        self.gpu_execution_time
    }
}
