mod arena_allocator;
mod bucketed_allocator;
mod buffer_cache;
mod caching_allocator;
mod direct_allocator;
mod error;
mod tiered_allocator;
mod trace;
mod uzu_allocator;

pub use arena_allocator::ArenaAllocator;
pub use bucketed_allocator::BucketedAllocator;
pub use buffer_cache::BufferCache;
pub use caching_allocator::CachingAllocator;
pub use direct_allocator::DirectAllocator;
pub use error::AllocError;
pub use tiered_allocator::TieredAllocator;
pub use trace::{AllocationEvent, AllocationPhase, AllocationSummary, AllocationTracer};
pub use uzu_allocator::UzuAllocator;

use super::{MTLBuffer, MTLResourceOptions, ProtocolObject, Retained};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferLifetime {
    Permanent,
    Scratch,
}

pub struct AllocatedBuffer {
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub size: usize,
    pub lifetime: BufferLifetime,
}

impl AllocatedBuffer {
    pub fn new(
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        size: usize,
        lifetime: BufferLifetime,
    ) -> Self {
        Self {
            buffer,
            size,
            lifetime,
        }
    }
}

pub trait MetalAllocator: Send + Sync {
    fn alloc_permanent(
        &self,
        size: usize,
    ) -> Result<AllocatedBuffer, AllocError>;

    fn alloc_scratch(
        &self,
        size: usize,
    ) -> Result<AllocatedBuffer, AllocError>;

    fn free(
        &self,
        buffer: AllocatedBuffer,
    );

    #[allow(unused_variables)]
    fn set_scratch_pool_limit(
        &self,
        size: usize,
    ) -> usize {
        0
    }

    fn active_memory(&self) -> usize;

    fn peak_memory(&self) -> usize;

    fn reset_peak_memory(&self) {}

    fn cache_memory(&self) -> usize {
        0
    }

    fn clear_cache(&self) {}

    fn resource_options(&self) -> MTLResourceOptions {
        MTLResourceOptions::STORAGE_MODE_SHARED
    }
}
