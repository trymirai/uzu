use std::cell::RefCell;

use super::{command_buffer::CpuCommandBuffer, context::CpuContext, error::CpuError, kernel::CpuKernels};
use crate::backends::common::Backend;

#[derive(Debug, Clone)]
pub struct Cpu;

impl Backend for Cpu {
    type Context = CpuContext;
    type CommandBuffer = CpuCommandBuffer;
    type Buffer = Box<[u8]>;
    type Event = RefCell<u64>;
    type Kernels = CpuKernels;
    type Error = CpuError;

    const MAX_INLINE_BYTES: usize = usize::MAX;
}
