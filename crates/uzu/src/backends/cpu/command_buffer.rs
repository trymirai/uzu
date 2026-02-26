use std::{cell::RefCell, time::Instant};

use super::Cpu;
use crate::backends::common::{CommandBuffer, CopyEncoder};

pub struct CpuCommandBuffer {
    commands: Vec<Box<dyn Fn()>>,
    completion_handlers: Vec<Box<dyn Fn()>>,
    gpu_execution_time_ms: Option<f64>,
}

impl CpuCommandBuffer {
    pub fn new() -> RefCell<CpuCommandBuffer> {
        RefCell::new(CpuCommandBuffer {
            commands: Vec::new(),
            completion_handlers: Vec::new(),
            gpu_execution_time_ms: None,
        })
    }

    pub fn push_command(
        &mut self,
        command: impl Fn() + 'static,
    ) {
        self.commands.push(Box::new(command))
    }
}

impl CommandBuffer for RefCell<CpuCommandBuffer> {
    type Backend = Cpu;

    fn with_compute_encoder<T>(
        &mut self,
        callback: impl FnOnce(&mut RefCell<CpuCommandBuffer>) -> T,
    ) -> T {
        callback(self)
    }

    fn with_copy_encoder<T>(
        &mut self,
        callback: impl FnOnce(&mut RefCell<CpuCommandBuffer>) -> T,
    ) -> T {
        callback(self)
    }

    fn encode_wait_for_event(
        &mut self,
        event: &RefCell<u64>,
        value: u64,
    ) {
        let event = event.as_ptr();
        self.borrow_mut().push_command(Box::new(move || unsafe { assert!(*event >= value, "deadlock") }))
    }

    fn encode_signal_event(
        &mut self,
        event: &RefCell<u64>,
        value: u64,
    ) {
        let event = event.as_ptr();
        self.borrow_mut().push_command(Box::new(move || unsafe { *event = value }))
    }

    fn add_completion_handler(
        &mut self,
        handler: impl Fn() + 'static,
    ) {
        self.borrow_mut().completion_handlers.push(Box::new(move || handler()))
    }

    fn submit(&self) {
        let start = Instant::now();
        for command in &self.borrow().commands {
            command()
        }
        self.borrow_mut().gpu_execution_time_ms = Some((Instant::now() - start).as_secs_f64() * 1000.0);
        for completion_handler in self.borrow().completion_handlers.iter() {
            completion_handler();
        }
    }

    fn wait_until_completed(&self) {}

    fn gpu_execution_time_ms(&self) -> Option<f64> {
        self.borrow().gpu_execution_time_ms
    }
}

impl CopyEncoder for RefCell<CpuCommandBuffer> {
    type Backend = Cpu;

    fn encode_copy(
        &mut self,
        src: &Box<[u8]>,
        dst: &mut Box<[u8]>,
        size: usize,
    ) {
        let src = src.as_ptr();
        let dst = dst.as_ptr() as *mut u8;
        self.borrow_mut().push_command(Box::new(move || unsafe {
            std::ptr::copy(src, dst, size);
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
        self.borrow_mut().push_command(Box::new(move || unsafe {
            dst.write_bytes(value, size);
        }))
    }
}
