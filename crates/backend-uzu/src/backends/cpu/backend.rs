use std::{any::Any, cell::UnsafeCell, pin::Pin, sync::atomic::AtomicU64};

use super::{
    command_buffer::CpuCommandBuffer, context::CpuContext, error::CpuError, kernel::CpuKernels, sparse::CpuSparseBuffer,
};
use crate::backends::common::{Backend, Buffer};

#[derive(Debug, Clone)]
pub struct Cpu;

impl Cpu {
    pub fn buffer_downcast<B: Buffer<Backend = Self>>(buffer: &B) -> &UnsafeCell<Pin<Box<[u8]>>> {
        let buffer = buffer as &dyn Any;
        if let Some(buffer) = buffer.downcast_ref::<<Self as Backend>::DenseBuffer>() {
            buffer
        } else {
            unreachable!("Unsupported Cpu buffer type")
        }
    }
}

impl Backend for Cpu {
    type Context = CpuContext;
    type CommandBuffer = CpuCommandBuffer;
    type DenseBuffer = UnsafeCell<Pin<Box<[u8]>>>;
    type SparseBuffer = CpuSparseBuffer;
    type Event = Pin<Box<AtomicU64>>;
    type Kernels = CpuKernels;
    type Error = CpuError;

    const MIN_ALLOCATION_ALIGNMENT: usize = 64;
    const MAX_INLINE_BYTES: usize = usize::MAX;
}
