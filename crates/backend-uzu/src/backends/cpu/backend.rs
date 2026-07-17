use std::{
    cell::UnsafeCell,
    pin::Pin,
    sync::{Arc, Condvar, Mutex, PoisonError},
    time::Duration,
};

use crate::backends::{
    common::{Backend, SharedEvent},
    cpu::{
        command_buffer::CpuCommandBuffer, context::CpuContext, error::CpuError, kernel::CpuKernels,
        sparse::CpuSparseBuffer,
    },
};

#[derive(Debug, Clone)]
pub struct Cpu;

#[derive(Debug, Clone, Default)]
pub struct CpuSharedEvent(Arc<(Mutex<u64>, Condvar)>);

impl SharedEvent for CpuSharedEvent {
    fn signaled_value(&self) -> u64 {
        *self.0.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn wait_until_signaled_value_timeout_ms(
        &self,
        value: u64,
        timeout_ms: u64,
    ) -> bool {
        let (current, changed) = &*self.0;
        let current = current.lock().unwrap_or_else(PoisonError::into_inner);
        let (current, _) = changed
            .wait_timeout_while(current, Duration::from_millis(timeout_ms), |current| *current < value)
            .unwrap_or_else(PoisonError::into_inner);
        *current >= value
    }

    fn signal(
        &self,
        value: u64,
    ) {
        let (current, changed) = &*self.0;
        let mut current = current.lock().unwrap_or_else(PoisonError::into_inner);
        if value > *current {
            *current = value;
            changed.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use proc_macros::uzu_test;

    use crate::backends::{common::SharedEvent, cpu::CpuSharedEvent};

    #[uzu_test]
    fn shared_event_signal_is_monotonic() {
        let event = CpuSharedEvent::default();
        event.signal(2);
        event.signal(1);
        assert_eq!(event.signaled_value(), 2);
    }
}

impl Backend for Cpu {
    type Context = CpuContext;
    type CommandBuffer = CpuCommandBuffer;
    type DenseBuffer = UnsafeCell<Pin<Box<[u8]>>>;
    type SparseBuffer = CpuSparseBuffer;
    type Kernels = CpuKernels;
    type Error = CpuError;
    type SharedEvent = CpuSharedEvent;

    const MIN_ALLOCATION_ALIGNMENT: usize = 4;
    const MAX_ALLOCATION_ALIGNMENT: usize = 64;
    const ALLOCATION_GRANULARITY: usize = 8 * 1024 * 1024;
    const MAX_INLINE_BYTES: usize = usize::MAX;
}
