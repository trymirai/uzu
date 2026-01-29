use std::rc::Rc;

use uzu::backends::{
    common::{Buffer, BufferLifetime, Context},
    metal::{MTLContext, Metal},
};

use super::AllocatorTrait;

pub struct CachingAllocator {
    context: Rc<MTLContext>,
}

impl CachingAllocator {
    pub fn new() -> Self {
        let context = MTLContext::new().expect("Failed to create MTLContext");
        Self { context }
    }
}

impl AllocatorTrait for CachingAllocator {
    type Buffer = Buffer<Metal>;

    fn alloc(&self, size: usize) -> Self::Buffer {
        self.context
            .allocator()
            .alloc(BufferLifetime::Scratch, size)
            .expect("Allocation failed")
    }

    fn free(&self, buffer: Self::Buffer) {
        drop(buffer);
    }

    fn peak_memory(&self) -> usize {
        self.context.allocator().peak_memory()
    }

    fn cache_memory(&self) -> usize {
        self.context.allocator().cache_memory()
    }
}
