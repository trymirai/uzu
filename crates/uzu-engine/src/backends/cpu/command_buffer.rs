use std::{
    sync::{Arc, mpsc},
    time::{Duration, Instant},
};

use crate::{
    backends::{
        common::{
            Allocation, AllocationPool, AllocationType, Buffer, BufferRangeMut, BufferRangeRef, CommandBuffer,
            CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
        },
        cpu::{Cpu, context::CpuContext, error::CpuError},
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
    completion_handlers: Vec<Box<dyn FnOnce(Result<&CpuCommandBufferCompleted, CpuError>) + Send + 'static>>,
    context: Arc<CpuContext>,
    allocation_pool: Arc<AllocationPool<Cpu>>,
}

impl CpuCommandBufferEncoding {
    pub fn new(
        context: Arc<CpuContext>,
        allocation_pool: Arc<AllocationPool<Cpu>>,
    ) -> CpuCommandBufferEncoding {
        CpuCommandBufferEncoding {
            commands: Vec::new(),
            completion_handlers: Vec::new(),
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
    ) -> Result<Allocation<Cpu>, CpuError> {
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
    ) -> Result<Allocation<Cpu>, CpuError> {
        self.context.allocator.allocate(
            size,
            AllocationType::Pooled {
                pool: &self.allocation_pool,
                cpu_available: false,
            },
        )
    }

    fn encode_copy<Src: Buffer<Backend = Cpu>, Dst: Buffer<Backend = Cpu>>(
        &mut self,
        src: BufferRangeRef<Src>,
        dst: BufferRangeMut<Dst>,
    ) {
        let src_range = src.range();
        let dst_range = dst.range();
        assert_eq!(src_range.len(), dst_range.len());

        let src_buffer = (src.buffer() as &dyn Buffer<Backend = Cpu>).downcast();
        let src_ptr = SendPtr(unsafe { (&*src_buffer.get()).as_ptr().add(src_range.start) });

        let dst_buffer = (dst.buffer() as &dyn Buffer<Backend = Cpu>).downcast();
        let dst_ptr = SendPtrMut(unsafe { (&mut *dst_buffer.get()).as_mut_ptr().add(dst_range.start) });

        self.push_command(move || unsafe {
            std::ptr::copy(src_ptr.as_ptr(), dst_ptr.as_ptr(), src_range.len());
        });
    }

    fn encode_fill<Dst: Buffer<Backend = Cpu>>(
        &mut self,
        dst: BufferRangeMut<Dst>,
        value: u8,
    ) {
        let range = dst.range();
        let size = range.end - range.start;
        let dst = SendPtrMut(unsafe {
            (&mut *(dst.buffer() as &dyn Buffer<Backend = Cpu>).downcast().get()).as_mut_ptr().add(range.start)
        });
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
            completion_handlers: self.completion_handlers,
            context: self.context,
            allocation_pool: self.allocation_pool,
        }
    }
}

pub struct CpuCommandBufferExecutable {
    commands: Vec<Box<dyn FnOnce() + Send>>,
    completion_handlers: Vec<Box<dyn FnOnce(Result<&CpuCommandBufferCompleted, CpuError>) + Send + 'static>>,
    context: Arc<CpuContext>,
    allocation_pool: Arc<AllocationPool<Cpu>>,
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

                for handler in self.completion_handlers {
                    handler(Ok(&completed))
                }

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
    _allocation_pool: Arc<AllocationPool<Cpu>>,
}

impl CommandBufferCompleted for CpuCommandBufferCompleted {
    type CommandBuffer = CpuCommandBuffer;

    fn gpu_execution_time(&self) -> Duration {
        self.gpu_execution_time
    }
}
