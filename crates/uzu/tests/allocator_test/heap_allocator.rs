use std::cell::Cell;

use metal::{MTLBuffer, MTLDevice, MTLDeviceExt, MTLHeap, MTLHeapDescriptor, MTLResourceOptions, MTLStorageMode};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::allocator_trait::AllocatorTrait;

pub struct MTLHeapAllocator {
    heap: Retained<ProtocolObject<dyn MTLHeap>>,
    peak_used: Cell<usize>,
    active_memory: Cell<usize>,
}

impl MTLHeapAllocator {
    pub fn new(heap_size: usize) -> Self {
        let device = <dyn MTLDevice>::system_default().expect("No Metal device");

        let descriptor = MTLHeapDescriptor::new();
        descriptor.set_size(heap_size);
        descriptor.set_storage_mode(MTLStorageMode::Shared);

        let heap = device.new_heap_with_descriptor(&descriptor).expect("Failed to create heap");

        Self {
            heap,
            peak_used: Cell::new(0),
            active_memory: Cell::new(0),
        }
    }
}

impl AllocatorTrait for MTLHeapAllocator {
    type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

    fn alloc(
        &self,
        size: usize,
    ) -> Self::Buffer {
        let buffer = self
            .heap
            .new_buffer(size, MTLResourceOptions::STORAGE_MODE_SHARED)
            .expect("Failed to create buffer from heap");

        let buf_size = buffer.length();
        let new_active = self.active_memory.get() + buf_size;
        self.active_memory.set(new_active);

        let used = self.heap.used_size();
        if used > self.peak_used.get() {
            self.peak_used.set(used);
        }

        buffer
    }

    fn free(
        &self,
        buffer: Self::Buffer,
    ) {
        let size = buffer.length();
        self.active_memory.set(self.active_memory.get().saturating_sub(size));
        drop(buffer);
    }

    fn peak_memory(&self) -> usize {
        self.peak_used.get()
    }

    fn cache_memory(&self) -> usize {
        0
    }

    fn active_memory(&self) -> usize {
        self.active_memory.get()
    }
}
