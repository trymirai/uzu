use std::{cell::UnsafeCell, pin::Pin};

use super::{
    command_buffer::CpuCommandBuffer, context::CpuContext, error::CpuError, kernel::CpuKernels, sparse::CpuSparseBuffer,
};
use crate::backends::common::Backend;

#[derive(Debug, Clone)]
pub struct Cpu;

impl Backend for Cpu {
    type Context = CpuContext;
    type CommandBuffer = CpuCommandBuffer;
    type DenseBuffer = UnsafeCell<Pin<Box<[u8]>>>;
    type SparseBuffer = CpuSparseBuffer;
    type Kernels = CpuKernels;
    type Error = CpuError;

    const MIN_ALLOCATION_ALIGNMENT: usize = 4;
    const MAX_ALLOCATION_ALIGNMENT: usize = 64;
    const ALLOCATION_GRANULARITY: usize = 8 * 1024 * 1024;
    const MAX_INLINE_BYTES: usize = usize::MAX;
}
