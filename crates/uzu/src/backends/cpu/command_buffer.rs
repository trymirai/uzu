use std::{
    cell::{OnceCell, RefCell},
    time::Instant,
};

use super::Cpu;
use crate::backends::common::{CommandBuffer, CopyEncoder};

pub struct CpuCommandBuffer {
    commands: Vec<Box<dyn Fn()>>,
    completion_handlers: Vec<Box<dyn FnOnce()>>,
    gpu_execution_time_ms: OnceCell<f64>,
}

impl CpuCommandBuffer {
    pub fn new() -> CpuCommandBuffer {
        CpuCommandBuffer {
            commands: Vec::new(),
            completion_handlers: Vec::new(),
            gpu_execution_time_ms: OnceCell::new(),
        }
    }

    pub fn push_command(
        &mut self,
        command: impl Fn() + 'static,
    ) {
        self.commands.push(Box::new(command))
    }
}

impl CommandBuffer for CpuCommandBuffer {
    type Backend = Cpu;

    fn with_compute_encoder<T>(
        &mut self,
        callback: impl FnOnce(&mut CpuCommandBuffer) -> T,
    ) -> T {
        callback(self)
    }

    fn with_copy_encoder<T>(
        &mut self,
        callback: impl FnOnce(&mut CpuCommandBuffer) -> T,
    ) -> T {
        callback(self)
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
        handler: impl FnOnce() + 'static,
    ) {
        self.completion_handlers.push(Box::new(handler))
    }

    fn submit(&mut self) {
        let start = Instant::now();
        for command in &self.commands {
            command()
        }
        self.gpu_execution_time_ms
            .set((Instant::now() - start).as_secs_f64() * 1000.0)
            .expect("gpu execution time already set");
        for completion_handler in self.completion_handlers.drain(..) {
            completion_handler();
        }
    }

    fn wait_until_completed(&self) {}

    fn gpu_execution_time_ms(&self) -> Option<f64> {
        self.gpu_execution_time_ms.get().copied()
    }
}

impl CopyEncoder for CpuCommandBuffer {
    type Backend = Cpu;

    fn encode_copy(
        &mut self,
        src: &Box<[u8]>,
        dst: &mut Box<[u8]>,
        size: usize,
    ) {
        let src = src.as_ptr();
        let dst = dst.as_ptr() as *mut u8;
        self.push_command(Box::new(move || unsafe {
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
        self.push_command(Box::new(move || unsafe {
            dst.write_bytes(value, size);
        }))
    }
}
