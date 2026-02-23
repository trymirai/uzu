use std::{
    cell::{Cell, RefCell},
    ops::Range,
    ptr,
    sync::Arc,
    time::{Duration, Instant},
};

use crate::backends::{
    common::{Backend, CommandBuffer, CopyEncoder, NativeBuffer},
    cpu::backend::Cpu,
};

#[derive(Clone)]
pub struct CpuCommandBuffer {
    buffer: Arc<RefCell<Vec<Box<dyn Fn()>>>>,
    completion_handlers: Arc<RefCell<Vec<Box<dyn Fn()>>>>,
    execution_duration: Cell<Option<Duration>>,
}

impl CpuCommandBuffer {
    pub fn new() -> Self {
        Self {
            buffer: Arc::new(RefCell::new(Vec::new())),
            completion_handlers: Arc::new(RefCell::new(Vec::new())),
            execution_duration: Cell::new(None),
        }
    }

    fn push(
        &self,
        closure: Box<dyn Fn()>,
    ) {
        self.buffer.borrow_mut().push(closure);
    }
}

impl CommandBuffer for CpuCommandBuffer {
    type Backend = Cpu;

    fn with_compute_encoder<T>(
        &self,
        callback: impl FnOnce(&<Self::Backend as Backend>::ComputeEncoder) -> T,
    ) -> T {
        callback(self)
    }

    fn with_copy_encoder<T>(
        &self,
        callback: impl FnOnce(&<Self::Backend as Backend>::CopyEncoder) -> T,
    ) -> T {
        callback(self)
    }

    fn encode_wait_for_event(
        &self,
        event: &<Self::Backend as Backend>::Event,
        value: u64,
    ) {
        let ev = event.clone();
        self.push(Box::new(move || {
            assert!(*ev.borrow() >= value, "deadlock!");
        }))
    }

    fn encode_signal_event(
        &self,
        event: &<Self::Backend as Backend>::Event,
        value: u64,
    ) {
        let ev = event.clone();
        self.push(Box::new(move || {
            assert!(*ev.borrow() <= value, "attempt to decrease event value");
            *ev.borrow_mut() = value;
        }))
    }

    fn add_completed_handler(
        &self,
        handler: impl Fn() + 'static,
    ) {
        self.completion_handlers.borrow_mut().push(Box::new(handler));
    }

    fn submit(&self) {
        let start = Instant::now();
        for task in self.buffer.borrow().iter() {
            task();
        }
        self.execution_duration.set(Some(start.elapsed()));

        for completion_handler in self.completion_handlers.borrow().iter() {
            completion_handler();
        }
    }

    fn wait_until_completed(&self) {}

    fn gpu_execution_time_ms(&self) -> Option<f64> {
        self.execution_duration.get().map(|duration| duration.as_millis() as f64)
    }
}

impl CopyEncoder for CpuCommandBuffer {
    type Backend = Cpu;

    fn encode_copy(
        &self,
        src: &<Self::Backend as Backend>::NativeBuffer,
        dst: &<Self::Backend as Backend>::NativeBuffer,
        size: usize,
    ) {
        assert!(src.length() >= size && dst.length() >= size);
        unsafe { ptr::copy_nonoverlapping(src.cpu_ptr().as_ptr(), dst.cpu_ptr().as_ptr(), size) }
    }

    fn encode_fill(
        &self,
        dst: &<Self::Backend as Backend>::NativeBuffer,
        range: Range<usize>,
        value: u8,
    ) {
        assert!(range.start <= range.end && range.end <= dst.length());
        unsafe {
            ptr::write_bytes((dst.cpu_ptr().as_ptr() as *mut u8).add(range.start), value, range.end - range.start);
        }
    }
}
