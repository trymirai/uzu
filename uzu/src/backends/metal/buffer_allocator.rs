use metal::{Buffer, Device, Heap, MTLResourceOptions};

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
