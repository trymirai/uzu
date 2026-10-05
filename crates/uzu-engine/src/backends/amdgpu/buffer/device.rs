use std::{collections::HashMap, mem::ManuallyDrop, os::raw::c_void, ptr::NonNull, sync::Arc};

use parking_lot::Mutex;

use super::dense::{track_allocation, track_release};
use crate::backends::{
    amdgpu::{
        Amdgpu,
        error::AmdgpuError,
        hip::{HipStream, hip, hip_call},
    },
    common::Buffer,
};

/// Device-local memory (`hipMalloc`) without a CPU mapping, for scratch allocations that only kernels touch.
/// On the APU it is the same DRAM as pinned host memory, but the driver maps it for the GPU alone: atomics,
/// copies and fills on it run at device speed, while on `hipHostMalloc` memory they go through the host path.
#[derive(Debug)]
pub struct AmdgpuDeviceBuffer {
    ptr: NonNull<c_void>,
    size: usize,
}

impl AmdgpuDeviceBuffer {
    /// A zero-filled allocation; the fill is queued on `stream`, ahead of the commands that will use the buffer.
    pub(in crate::backends::amdgpu) fn new(
        size: usize,
        stream: HipStream,
    ) -> Result<Self, AmdgpuError> {
        let hip = hip()?;
        let allocation_size = size.max(64);
        let mut ptr = std::ptr::null_mut();
        let started = std::time::Instant::now();
        hip_call!(hip, hipMalloc(&mut ptr, allocation_size))?;
        let ptr = NonNull::new(ptr).ok_or(AmdgpuError::Hip {
            call: "hipMalloc",
            code: 0,
            message: "returned a null pointer".into(),
        })?;
        // Same zero-fill as the host buffers, so that results do not depend on where a scratch page came from.
        // It must go to the stream that uses the page: hipMemset runs on the legacy null stream, which a
        // non-blocking stream does not wait for, and when the null stream was busy (contexts of tests running in
        // parallel) the fill landed after the first kernels had written the page.
        hip_call!(hip, hipMemsetAsync(ptr.as_ptr(), 0, allocation_size, stream))?;
        crate::backends::amdgpu::profile::record_memory_call("hipMalloc+hipMemsetAsync", started.elapsed());
        track_allocation(allocation_size);
        Ok(Self {
            ptr,
            size,
        })
    }

    pub(in crate::backends::amdgpu) fn device_address(&self) -> u64 {
        self.ptr.as_ptr() as u64
    }
}

// The allocation has a stable address; callers synchronize access through buffer borrows and command buffers.
unsafe impl Send for AmdgpuDeviceBuffer {}
unsafe impl Sync for AmdgpuDeviceBuffer {}

impl Drop for AmdgpuDeviceBuffer {
    fn drop(&mut self) {
        if let Ok(hip) = hip() {
            let started = std::time::Instant::now();
            let _ = hip_call!(hip, hipFree(self.ptr.as_ptr()));
            crate::backends::amdgpu::profile::record_memory_call("hipFree", started.elapsed());
        }
        track_release(self.size.max(64));
    }
}

impl Buffer for AmdgpuDeviceBuffer {
    type Backend = Amdgpu;

    fn size(&self) -> usize {
        self.size
    }
}

/// Free scratch pages by size, shared by every allocation pool of a context. The engine creates a new pool for
/// each decode step and drops the previous one; with pages of their own every step paid hipMalloc + zero-fill
/// per page and a hipFree per page that waits for all queued GPU work (thousands per generation, seconds of
/// CPU blocked on the GPU). Pages of a dropped pool come back here instead.
pub(in crate::backends::amdgpu) type ScratchPageCache = Arc<Mutex<HashMap<usize, Vec<AmdgpuDeviceBuffer>>>>;

/// A scratch pool page that returns to the context's cache when its pool drops it.
#[derive(Debug)]
pub struct ScratchPage {
    buffer: ManuallyDrop<AmdgpuDeviceBuffer>,
    cache: ScratchPageCache,
}

impl ScratchPage {
    pub(in crate::backends::amdgpu) fn take(
        cache: &ScratchPageCache,
        size: usize,
        stream: HipStream,
    ) -> Result<Self, AmdgpuError> {
        let cached = cache.lock().get_mut(&size).and_then(Vec::pop);
        let buffer = match cached {
            Some(buffer) => buffer,
            None => AmdgpuDeviceBuffer::new(size, stream)?,
        };
        Ok(Self {
            buffer: ManuallyDrop::new(buffer),
            cache: cache.clone(),
        })
    }

    pub(in crate::backends::amdgpu) fn device_address(&self) -> u64 {
        self.buffer.device_address()
    }
}

impl Drop for ScratchPage {
    fn drop(&mut self) {
        let buffer = unsafe { ManuallyDrop::take(&mut self.buffer) };
        self.cache.lock().entry(buffer.size).or_default().push(buffer);
    }
}

impl Buffer for ScratchPage {
    type Backend = Amdgpu;

    fn size(&self) -> usize {
        self.buffer.size
    }
}

#[cfg(test)]
#[path = "../../../../unit/backends/amdgpu/buffer/scratch_page_test.rs"]
mod tests;
