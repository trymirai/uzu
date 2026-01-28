use std::sync::atomic::{AtomicUsize, Ordering};

use super::{
    super::{
        MTLBuffer, MTLDevice, MTLDeviceExt, MTLResourceOptions, ProtocolObject,
        Retained,
    },
    AllocError, AllocatedBuffer, BufferLifetime, MetalAllocator,
};

pub struct DirectAllocator {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    active_memory: AtomicUsize,
    peak_memory: AtomicUsize,
    resource_options: MTLResourceOptions,
}

// SAFETY: Metal objects on Apple Silicon are thread-safe.
// The device has its own internal synchronization.
unsafe impl Send for DirectAllocator {}
unsafe impl Sync for DirectAllocator {}

impl DirectAllocator {
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            device,
            active_memory: AtomicUsize::new(0),
            peak_memory: AtomicUsize::new(0),
            resource_options: MTLResourceOptions::STORAGE_MODE_SHARED,
        }
    }

    pub fn with_resource_options(
        mut self,
        options: MTLResourceOptions,
    ) -> Self {
        self.resource_options = options;
        self
    }

    fn allocate_buffer(
        &self,
        size: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, AllocError> {
        self.device.new_buffer(size, self.resource_options).ok_or_else(|| {
            AllocError::AllocationFailed {
                size,
                reason: "device.newBuffer returned nil".to_string(),
            }
        })
    }

    fn track_allocation(
        &self,
        size: usize,
    ) {
        let new_active =
            self.active_memory.fetch_add(size, Ordering::Relaxed) + size;
        self.peak_memory.fetch_max(new_active, Ordering::Relaxed);
    }

    fn track_deallocation(
        &self,
        size: usize,
    ) {
        self.active_memory.fetch_sub(size, Ordering::Relaxed);
    }
}

impl MetalAllocator for DirectAllocator {
    fn alloc_permanent(
        &self,
        size: usize,
    ) -> Result<AllocatedBuffer, AllocError> {
        let buffer = self.allocate_buffer(size)?;
        self.track_allocation(size);
        Ok(AllocatedBuffer::new(buffer, size, BufferLifetime::Permanent))
    }

    fn alloc_scratch(
        &self,
        size: usize,
    ) -> Result<AllocatedBuffer, AllocError> {
        let buffer = self.allocate_buffer(size)?;
        self.track_allocation(size);
        Ok(AllocatedBuffer::new(buffer, size, BufferLifetime::Scratch))
    }

    fn free(
        &self,
        buffer: AllocatedBuffer,
    ) {
        self.track_deallocation(buffer.size);
        drop(buffer);
    }

    fn active_memory(&self) -> usize {
        self.active_memory.load(Ordering::Relaxed)
    }

    fn peak_memory(&self) -> usize {
        self.peak_memory.load(Ordering::Relaxed)
    }

    fn reset_peak_memory(&self) {
        self.peak_memory.store(
            self.active_memory.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }

    fn resource_options(&self) -> MTLResourceOptions {
        self.resource_options
    }
}
