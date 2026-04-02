use std::{cell::UnsafeCell, pin::Pin, sync::atomic::AtomicU64};

use super::{command_buffer::CpuCommandBuffer, context::CpuContext, error::CpuError, kernel::CpuKernels};
use crate::backends::common::Backend;

#[derive(Debug, Clone)]
pub struct Cpu;

impl Backend for Cpu {
    type Context = CpuContext;
    type CommandBuffer = CpuCommandBuffer;
    type Buffer = UnsafeCell<Pin<Box<[u8]>>>;
    type Event = Pin<Box<AtomicU64>>;
    type Kernels = CpuKernels;
    type Error = CpuError;

    const MIN_ALLOCATION_ALIGNMENT: usize = 64;
    const MAX_INLINE_BYTES: usize = usize::MAX;
}
