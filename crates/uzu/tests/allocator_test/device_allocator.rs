use std::cell::Cell;

use metal::{MTLBuffer, MTLDevice, MTLDeviceExt, MTLResourceOptions};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::allocator_trait::AllocatorTrait;

pub struct DeviceAllocator {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    active_memory: Cell<usize>,
    peak_memory: Cell<usize>,
}

impl DeviceAllocator {
    pub fn new() -> Self {
        let device =
            <dyn MTLDevice>::system_default().expect("No Metal device");

        Self {
            device,
            active_memory: Cell::new(0),
            peak_memory: Cell::new(0),
        }
    }
}

impl AllocatorTrait for DeviceAllocator {
    type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

    fn alloc(
        &self,
        size: usize,
    ) -> Self::Buffer {
        let buffer = self
            .device
            .new_buffer(size, MTLResourceOptions::STORAGE_MODE_SHARED)
            .expect("Failed to create buffer");

        let buf_size = buffer.length();
        let new_active = self.active_memory.get() + buf_size;
        self.active_memory.set(new_active);

        if new_active > self.peak_memory.get() {
            self.peak_memory.set(new_active);
        }

        buffer
    }

    fn free(
        &self,
        buffer: Self::Buffer,
    ) {
        let size = buffer.length();
        self.active_memory
            .set(self.active_memory.get().saturating_sub(size));
        drop(buffer);
    }

    fn peak_memory(&self) -> usize {
        self.peak_memory.get()
    }

    fn cache_memory(&self) -> usize {
        0
    }

    fn active_memory(&self) -> usize {
        self.active_memory.get()
    }
}
