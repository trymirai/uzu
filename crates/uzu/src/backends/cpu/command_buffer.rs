use std::{
    cell::UnsafeCell,
    pin::Pin,
    sync::atomic::{AtomicU64, Ordering},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crate::{
    backends::{
        common::{
            AccessFlags, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable,
            CommandBufferInitial, CommandBufferPending,
        },
        cpu::{Cpu, error::CpuError},
    },
    utils::pointers::{SendPtr, SendPtrMut},
};

pub struct CpuCommandBuffer;

impl CommandBuffer for CpuCommandBuffer {
    type Backend = Cpu;

    type Initial = CpuCommandBufferInitial;
    type Encoding = CpuCommandBufferEncoding;
    type Executable = CpuCommandBufferExecutable;
    type Pending = CpuCommandBufferPending;
    type Completed = CpuCommandBufferCompleted;
}

pub struct CpuCommandBufferInitial;

impl CpuCommandBufferInitial {
    pub fn new() -> CpuCommandBufferInitial {
        CpuCommandBufferInitial
    }
}

impl CommandBufferInitial for CpuCommandBufferInitial {
    type CommandBuffer = CpuCommandBuffer;

    fn start_encoding(self) -> CpuCommandBufferEncoding {
        CpuCommandBufferEncoding {
            commands: Vec::new(),
            completion_handlers: Vec::new(),
        }
    }
}

pub struct CpuCommandBufferEncoding {
    commands: Vec<Box<dyn FnOnce() + Send>>,
    completion_handlers: Vec<Box<dyn FnOnce(Result<&CpuCommandBufferCompleted, CpuError>) + Send + 'static>>,
}

impl CpuCommandBufferEncoding {
    pub fn push_command(
        &mut self,
        command: impl FnOnce() + Send + 'static,
    ) {
        self.commands.push(Box::new(command))
    }
}

impl CommandBufferEncoding for CpuCommandBufferEncoding {
    type CommandBuffer = CpuCommandBuffer;

    fn encode_copy(
        &mut self,
        src: &UnsafeCell<Pin<Box<[u8]>>>,
        src_range: std::ops::Range<usize>,
        dst: &mut UnsafeCell<Pin<Box<[u8]>>>,
        dst_range: std::ops::Range<usize>,
    ) {
        let size = src_range.end - src_range.start;
        assert_eq!(size, dst_range.end - dst_range.start);
        assert!(unsafe { &*src.get() }.len() >= src_range.end);
        assert!(dst.get_mut().len() >= dst_range.end);

        let src_ptr = SendPtr(unsafe { (&*src.get()).as_ptr().add(src_range.start) });
        let dst_ptr = SendPtrMut(unsafe { dst.get_mut().as_mut_ptr().add(dst_range.start) });
        self.push_command(move || unsafe {
            std::ptr::copy(src_ptr.as_ptr(), dst_ptr.as_ptr(), size);
        });
    }

    fn encode_fill(
        &mut self,
        dst: &mut UnsafeCell<Pin<Box<[u8]>>>,
        range: std::ops::Range<usize>,
        value: u8,
    ) {
        let size = range.end - range.start;
        let dst = SendPtrMut(dst.get_mut()[range].as_ptr() as *mut u8);
        self.push_command(move || unsafe {
            dst.as_ptr().write_bytes(value, size);
        });
    }

    fn encode_barrier(
        &mut self,
        _after: AccessFlags,
        _before: AccessFlags,
    ) {
    }

    fn encode_wait_for_event(
        &mut self,
        event: &Pin<Box<AtomicU64>>,
        value: u64,
    ) {
        let event = SendPtr(event.as_ref().get_ref() as *const AtomicU64);
        self.push_command(move || unsafe {
            while (&*event.as_ptr()).load(Ordering::Acquire) < value {
                std::hint::spin_loop();
            }
        })
    }

    fn encode_signal_event(
        &mut self,
        event: &Pin<Box<AtomicU64>>,
        value: u64,
    ) {
        let event = SendPtr(event.as_ref().get_ref() as *const AtomicU64);
        self.push_command(Box::new(move || unsafe {
            (&*event.as_ptr()).store(value, Ordering::Release);
        }))
    }

    fn add_completion_handler(
        &mut self,
        handler: impl FnOnce(Result<&CpuCommandBufferCompleted, CpuError>) + Send + 'static,
    ) {
        self.completion_handlers.push(Box::new(handler))
    }

    fn end_encoding(self) -> CpuCommandBufferExecutable {
        CpuCommandBufferExecutable {
            commands: self.commands,
            completion_handlers: self.completion_handlers,
        }
    }
}

pub struct CpuCommandBufferExecutable {
    commands: Vec<Box<dyn FnOnce() + Send>>,
    completion_handlers: Vec<Box<dyn FnOnce(Result<&CpuCommandBufferCompleted, CpuError>) + Send + 'static>>,
}

impl CommandBufferExecutable for CpuCommandBufferExecutable {
    type CommandBuffer = CpuCommandBuffer;

    fn submit(self) -> CpuCommandBufferPending {
        let thread_handle = thread::spawn(move || {
            let start = Instant::now();

            for command in self.commands {
                command()
            }

            let gpu_execution_time = start.elapsed();

            let completed = CpuCommandBufferCompleted {
                gpu_execution_time,
            };

            for handler in self.completion_handlers {
                handler(Ok(&completed))
            }

            completed
        });

        CpuCommandBufferPending {
            thread_handle,
        }
    }
}

pub struct CpuCommandBufferPending {
    thread_handle: JoinHandle<CpuCommandBufferCompleted>,
}

impl CommandBufferPending for CpuCommandBufferPending {
    type CommandBuffer = CpuCommandBuffer;

    fn wait_until_completed(self) -> Result<CpuCommandBufferCompleted, CpuError> {
        self.thread_handle.join().map_err(CpuError::CommandBufferExecutionFailed)
    }
}

pub struct CpuCommandBufferCompleted {
    gpu_execution_time: Duration,
}

impl CommandBufferCompleted for CpuCommandBufferCompleted {
    type CommandBuffer = CpuCommandBuffer;

    fn gpu_execution_time(&self) -> Option<Duration> {
        Some(self.gpu_execution_time)
    }
}
