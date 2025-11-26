use std::cell::Cell;

use metal::{
    Buffer, Device, Heap, HeapDescriptor, MTLCPUCacheMode, MTLResourceOptions,
    MTLStorageMode,
};

pub trait BufferAllocator {
    fn allocate_buffer(
        &self,
        size_in_bytes: usize,
        options: MTLResourceOptions,
    ) -> Buffer;
}

impl BufferAllocator for Device {
    #[inline]
    fn allocate_buffer(
        &self,
        size_in_bytes: usize,
        options: MTLResourceOptions,
    ) -> Buffer {
        self.new_buffer(size_in_bytes as u64, options)
    }
}

impl BufferAllocator for Heap {
    #[inline]
    fn allocate_buffer(
        &self,
        size_in_bytes: usize,
        options: MTLResourceOptions,
    ) -> Buffer {
        self.new_buffer(size_in_bytes as u64, options).unwrap()
    }
}

pub struct HeapAllocator {
    heap: Heap,
    allocated_bytes: Cell<u64>,
    total_bytes: u64,
}

impl HeapAllocator {
    pub fn new(
        device: &Device,
        size_bytes: u64,
        storage_mode: MTLStorageMode,
    ) -> Self {
        let descriptor = HeapDescriptor::new();
        descriptor.set_size(size_bytes);
        descriptor.set_storage_mode(storage_mode);
        descriptor.set_cpu_cache_mode(MTLCPUCacheMode::DefaultCache);

        let heap = device.new_heap(&descriptor);
        eprintln!(
            "[HeapAllocator] Created heap: size={}MB, storage_mode={:?}",
            size_bytes / (1024 * 1024),
            storage_mode
        );

        Self {
            heap,
            allocated_bytes: Cell::new(0),
            total_bytes: size_bytes,
        }
    }

    pub fn allocate(
        &self,
        size_bytes: u64,
        options: MTLResourceOptions,
    ) -> Option<Buffer> {
        let buffer = self.heap.new_buffer(size_bytes, options)?;
        let new_total =
            self.allocated_bytes.get() + self.heap.current_allocated_size();
        self.allocated_bytes.set(new_total);
        Some(buffer)
    }

    pub fn used_bytes(&self) -> u64 {
        self.heap.used_size()
    }

    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    #[allow(dead_code)]
    pub fn available_bytes(&self) -> u64 {
        self.heap.max_available_size_with_alignment(16)
    }
}

impl BufferAllocator for HeapAllocator {
    fn allocate_buffer(
        &self,
        size_in_bytes: usize,
        options: MTLResourceOptions,
    ) -> Buffer {
        self.allocate(size_in_bytes as u64, options)
            .expect("HeapAllocator: out of memory")
    }
}

pub struct FallbackHeapAllocator {
    heap: HeapAllocator,
    device: Device,
    heap_alloc_count: Cell<usize>,
    device_alloc_count: Cell<usize>,
}

impl FallbackHeapAllocator {
    pub fn new(
        device: Device,
        heap_size_bytes: u64,
        storage_mode: MTLStorageMode,
    ) -> Self {
        let heap = HeapAllocator::new(&device, heap_size_bytes, storage_mode);
        Self {
            heap,
            device,
            heap_alloc_count: Cell::new(0),
            device_alloc_count: Cell::new(0),
        }
    }

    pub fn stats(&self) -> (usize, usize, u64, u64) {
        (
            self.heap_alloc_count.get(),
            self.device_alloc_count.get(),
            self.heap.used_bytes(),
            self.heap.total_bytes(),
        )
    }
}

impl BufferAllocator for FallbackHeapAllocator {
    fn allocate_buffer(
        &self,
        size_in_bytes: usize,
        options: MTLResourceOptions,
    ) -> Buffer {
        if let Some(buffer) = self.heap.allocate(size_in_bytes as u64, options)
        {
            self.heap_alloc_count
                .set(self.heap_alloc_count.get() + 1);
            buffer
        } else {
            eprintln!(
                "[FallbackHeapAllocator] Heap full, falling back to device allocation: {}KB",
                size_in_bytes / 1024
            );
            self.device_alloc_count
                .set(self.device_alloc_count.get() + 1);
            self.device.new_buffer(size_in_bytes as u64, options)
        }
    }
}
