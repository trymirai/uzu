use std::{
    cell::UnsafeCell,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use super::{
    command_buffer::CpuCommandBuffer, context::CpuContext, error::CpuError, kernel::CpuKernels, sparse::CpuSparseBuffer,
};
use crate::backends::common::{Backend, SharedEvent};

#[derive(Debug, Clone)]
pub struct Cpu;

#[derive(Debug, Clone, Default)]
pub struct CpuSharedEvent(Arc<AtomicU64>);

impl SharedEvent for CpuSharedEvent {
    fn signaled_value(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }

    fn signal(
        &self,
        value: u64,
    ) {
        self.0.fetch_max(value, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use proc_macros::uzu_test;

    use super::CpuSharedEvent;
    use crate::backends::common::SharedEvent;

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
