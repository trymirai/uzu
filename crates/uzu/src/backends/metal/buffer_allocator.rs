use metal::{MTLBuffer, MTLDevice, MTLDeviceExt, MTLHeap, MTLHeapExt, MTLResourceOptions};
use objc2::{rc::Retained, runtime::ProtocolObject};

/// Type alias for owned MTLBuffer
pub type MTLBufferObj = Retained<ProtocolObject<dyn MTLBuffer>>;

pub trait BufferAllocator {
    fn allocate_buffer(
        &self,
        size_in_bytes: usize,
        options: MTLResourceOptions,
    ) -> MTLBufferObj;
}

impl BufferAllocator for ProtocolObject<dyn MTLDevice> {
    #[inline]
    fn allocate_buffer(
        &self,
        size_in_bytes: usize,
        options: MTLResourceOptions,
    ) -> MTLBufferObj {
        self.new_buffer(size_in_bytes, options)
            .expect("Failed to allocate buffer")
    }
}

impl BufferAllocator for ProtocolObject<dyn MTLHeap> {
    #[inline]
    fn allocate_buffer(
        &self,
        size_in_bytes: usize,
        options: MTLResourceOptions,
    ) -> MTLBufferObj {
        self.new_buffer(size_in_bytes, options)
            .expect("Failed to allocate buffer from heap")
    }
}
