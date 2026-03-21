use std::{
    cell::{OnceCell, RefCell},
    time::{Duration, Instant},
};

use super::Cpu;
use crate::backends::{
    common::{
        CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial,
        CommandBufferPending,
    },
    cpu::error::CpuError,
};

pub struct CpuCommandBuffer {
    commands: Vec<Box<dyn Fn()>>,
    completion_handlers: Vec<Box<dyn FnOnce(Result<&CpuCommandBuffer, CpuError>)>>,
    gpu_execution_time: OnceCell<Duration>,
}

impl CommandBuffer for CpuCommandBuffer {
    type Backend = Cpu;

    type Initial = CpuCommandBuffer;
    type Encoding = CpuCommandBuffer;
    type Executable = CpuCommandBuffer;
    type Pending = CpuCommandBuffer;
    type Completed = CpuCommandBuffer;
}

impl CpuCommandBuffer {
    pub fn new() -> CpuCommandBuffer {
        CpuCommandBuffer {
            commands: Vec::new(),
            completion_handlers: Vec::new(),
            gpu_execution_time: OnceCell::new(),
        }
    }

    pub fn push_command(
        &mut self,
        command: impl Fn() + 'static,
    ) {
        self.commands.push(Box::new(command))
    }
}

impl CommandBufferInitial for CpuCommandBuffer {
    type CommandBuffer = CpuCommandBuffer;

    fn start_encoding(self) -> CpuCommandBuffer {
        self
    }
}

impl CommandBufferEncoding for CpuCommandBuffer {
    type CommandBuffer = CpuCommandBuffer;

    fn encode_copy(
        &mut self,
        src: &Box<[u8]>,
        src_range: std::ops::Range<usize>,
        dst: &mut Box<[u8]>,
        dst_range: std::ops::Range<usize>,
    ) {
        let size = src_range.end - src_range.start;
        assert_eq!(size, dst_range.end - dst_range.start);
        let src_ptr = unsafe { src.as_ptr().add(src_range.start) };
        let dst_ptr = unsafe { dst.as_ptr().add(dst_range.start) as *mut u8 };
        self.push_command(Box::new(move || unsafe {
            std::ptr::copy(src_ptr, dst_ptr, size);
        }))
    }

    fn encode_fill(
        &mut self,
        dst: &mut Box<[u8]>,
        range: std::ops::Range<usize>,
        value: u8,
    ) {
        let size = range.end - range.start;
        let dst = dst[range].as_ptr() as *mut u8;
        self.push_command(Box::new(move || unsafe {
            dst.write_bytes(value, size);
        }))
    }

    fn encode_wait_for_event(
        &mut self,
        event: &RefCell<u64>,
        value: u64,
    ) {
        let event = event.as_ptr();
        self.push_command(Box::new(move || unsafe { assert!(*event >= value, "deadlock") }))
    }

    fn encode_signal_event(
        &mut self,
        event: &RefCell<u64>,
        value: u64,
    ) {
        let event = event.as_ptr();
        self.push_command(Box::new(move || unsafe { *event = value }))
    }

    fn add_completion_handler(
        &mut self,
        handler: impl FnOnce(Result<&CpuCommandBuffer, CpuError>) + 'static,
    ) {
        self.completion_handlers.push(Box::new(handler))
    }

    fn end_encoding(self) -> CpuCommandBuffer {
        self
    }
}

impl CommandBufferExecutable for CpuCommandBuffer {
    type CommandBuffer = CpuCommandBuffer;

    fn submit(mut self) -> CpuCommandBuffer {
        let start = Instant::now();
        for command in &self.commands {
            command()
        }

        self.gpu_execution_time.set(Instant::now() - start).expect("gpu execution time already set");

        for completion_handler in self.completion_handlers.drain(..).collect::<Vec<_>>() {
            completion_handler(Ok(&self));
        }

        CpuCommandBuffer {
            commands: Vec::new(),
            completion_handlers: Vec::new(),
            gpu_execution_time: self.gpu_execution_time,
        }
    }
}

impl CommandBufferPending for CpuCommandBuffer {
    type CommandBuffer = CpuCommandBuffer;

    fn wait_until_completed(self) -> Result<CpuCommandBuffer, CpuError> {
        Ok(self)
    }
}

impl CommandBufferCompleted for CpuCommandBuffer {
    type CommandBuffer = CpuCommandBuffer;

    fn gpu_execution_time(&self) -> Option<Duration> {
        self.gpu_execution_time.get().copied()
    }
}
