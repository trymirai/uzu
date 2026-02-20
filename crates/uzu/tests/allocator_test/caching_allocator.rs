use std::rc::Rc;

use uzu::backends::{
    common::{Buffer, BufferLifetime, Context},
    metal::{Metal, MetalContext},
};

use super::allocator_trait::AllocatorTrait;

pub struct CachingAllocator {
    pub context: Rc<MetalContext>,
}

impl CachingAllocator {
    pub fn new() -> Self {
        let context = MetalContext::new().expect("Failed to create MetalContext");
        Self {
            context,
        }
    }
}

impl AllocatorTrait for CachingAllocator {
    type Buffer = Buffer<Metal>;

    fn alloc(
        &self,
        size: usize,
    ) -> Self::Buffer {
        self.context.allocator().alloc(BufferLifetime::Scratch, size).expect("Allocation failed")
    }

    fn free(
        &self,
        buffer: Self::Buffer,
    ) {
        drop(buffer);
    }

    fn peak_memory(&self) -> usize {
        self.context.allocator().peak_memory()
    }

    fn cache_memory(&self) -> usize {
        self.context.allocator().cache_memory()
    }

    fn active_memory(&self) -> usize {
        self.context.allocator().active_memory()
    }
}
