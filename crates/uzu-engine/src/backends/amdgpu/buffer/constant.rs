use std::{mem::ManuallyDrop, os::raw::c_void, ptr::NonNull, sync::Arc};

use parking_lot::Mutex;

use super::dense::AmdgpuBuffer;
use crate::backends::{
    amdgpu::{Amdgpu, error::AmdgpuError},
    common::{Buffer, BufferCpuAccessible, ConstantBuffer, allocator::bump::BumpAllocation},
};

/// Size of the pages command buffers bump-allocate their constants from.
pub(in crate::backends::amdgpu) const CONSTANT_PAGE_SIZE: usize = 256 * 1024;

/// Free constant pages of `CONSTANT_PAGE_SIZE`, shared by the command buffers of a context. Allocating pinned
/// memory takes milliseconds and freeing it waits for all queued GPU work (hipHostFree synchronizes the
/// device), so a page per command buffer stalled the pipelining of command buffers; pages now come back here
/// once their command buffer has completed.
pub(in crate::backends::amdgpu) type ConstantPagePool = Arc<Mutex<Vec<AmdgpuBuffer>>>;

/// A constant page; standard-size pages return to the pool when dropped.
#[derive(Debug)]
pub struct ConstantPage {
    buffer: ManuallyDrop<AmdgpuBuffer>,
    pool: Option<ConstantPagePool>,
}

impl ConstantPage {
    pub(in crate::backends::amdgpu) fn take(
        pool: &ConstantPagePool,
        size: usize,
    ) -> Result<Self, AmdgpuError> {
        if size != CONSTANT_PAGE_SIZE {
            return Ok(Self {
                buffer: ManuallyDrop::new(AmdgpuBuffer::new(size)?),
                pool: None,
            });
        }
        let buffer = match pool.lock().pop() {
            Some(buffer) => buffer,
            None => AmdgpuBuffer::new(size)?,
        };
        Ok(Self {
            buffer: ManuallyDrop::new(buffer),
            pool: Some(pool.clone()),
        })
    }

    pub(in crate::backends::amdgpu) fn device_address(&self) -> u64 {
        self.buffer.device_address()
    }

    fn cpu_ptr(&self) -> NonNull<c_void> {
        self.buffer.cpu_ptr()
    }
}

impl Drop for ConstantPage {
    fn drop(&mut self) {
        let buffer = unsafe { ManuallyDrop::take(&mut self.buffer) };
        match &self.pool {
            Some(pool) => pool.lock().push(buffer),
            None => drop(buffer),
        }
    }
}

impl Buffer for BumpAllocation<ConstantPage> {
    type Backend = Amdgpu;

    fn size(&self) -> usize {
        self.range().iter().len()
    }
}

impl BufferCpuAccessible for BumpAllocation<ConstantPage> {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        unsafe { self.page().cpu_ptr().byte_add(self.range().start) }
    }
}

impl ConstantBuffer for BumpAllocation<ConstantPage> {}
