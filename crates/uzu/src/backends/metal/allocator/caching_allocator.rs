use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

use metal::{MTLHeapDescriptor, MTLStorageMode};

use super::{
    super::{
        MTLBuffer, MTLDevice, MTLDeviceExt, MTLHeap, MTLResourceOptions,
        ProtocolObject, Retained,
    },
    AllocError, AllocatedBuffer, BufferCache, BufferLifetime, MetalAllocator,
};

const PAGE_SIZE: usize = 16384;
const SMALL_ALLOC_SIZE: usize = 256;
const SMALL_HEAP_SIZE: usize = 1 << 20; // 1 MB

pub struct CachingAllocator {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    buffer_cache: Mutex<BufferCache>,
    small_heap: Option<Retained<ProtocolObject<dyn MTLHeap>>>,
    active_memory: AtomicUsize,
    peak_memory: AtomicUsize,
    gc_limit: AtomicUsize,
    max_pool_size: AtomicUsize,
    resource_options: MTLResourceOptions,
}

// SAFETY: Metal objects on Apple Silicon are thread-safe.
// The device and heap have their own internal synchronization.
unsafe impl Send for CachingAllocator {}
unsafe impl Sync for CachingAllocator {}

impl CachingAllocator {
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        let recommended_max =
            device.recommended_max_working_set_size() as usize;
        let total_memory = Self::get_system_memory();

        let gc_limit = if recommended_max > 0 {
            (recommended_max * 95) / 100
        } else {
            (total_memory * 80) / 100
        };

        let small_heap = Self::create_small_heap(&device);

        Self {
            device,
            buffer_cache: Mutex::new(BufferCache::new(PAGE_SIZE)),
            small_heap,
            active_memory: AtomicUsize::new(0),
            peak_memory: AtomicUsize::new(0),
            gc_limit: AtomicUsize::new(gc_limit),
            max_pool_size: AtomicUsize::new(usize::MAX),
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

    fn get_system_memory() -> usize {
        #[cfg(target_os = "macos")]
        {
            let mut size: u64 = 0;
            let mut len = std::mem::size_of::<u64>();
            let name = c"hw.memsize";
            unsafe {
                libc::sysctlbyname(
                    name.as_ptr(),
                    &mut size as *mut u64 as *mut _,
                    &mut len,
                    std::ptr::null_mut(),
                    0,
                );
            }
            size as usize
        }
        #[cfg(not(target_os = "macos"))]
        {
            16 * 1024 * 1024 * 1024 // Default 16GB
        }
    }

    fn create_small_heap(
        device: &Retained<ProtocolObject<dyn MTLDevice>>
    ) -> Option<Retained<ProtocolObject<dyn MTLHeap>>> {
        let descriptor = MTLHeapDescriptor::new();
        descriptor.set_size(SMALL_HEAP_SIZE);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        device.new_heap_with_descriptor(&descriptor)
    }

    fn align_to_page(
        &self,
        size: usize,
    ) -> usize {
        (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
    }

    fn allocate_from_device(
        &self,
        size: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, AllocError> {
        if size < SMALL_ALLOC_SIZE {
            if let Some(ref heap) = self.small_heap {
                if let Some(buffer) =
                    heap.new_buffer(size, self.resource_options)
                {
                    return Ok(buffer);
                }
            }
        }

        self.device.new_buffer(size, self.resource_options).ok_or_else(|| {
            AllocError::AllocationFailed {
                size,
                reason: "device.newBuffer returned nil".to_string(),
            }
        })
    }

    fn maybe_gc(&self) {
        let active = self.active_memory.load(Ordering::Relaxed);
        let gc_limit = self.gc_limit.load(Ordering::Relaxed);

        if active > gc_limit {
            let to_free = active - gc_limit;
            if let Ok(mut cache) = self.buffer_cache.lock() {
                cache.release_cached_buffers(to_free);
            }
        }
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

    fn allocate_internal(
        &self,
        size: usize,
        lifetime: BufferLifetime,
    ) -> Result<AllocatedBuffer, AllocError> {
        let aligned_size = self.align_to_page(size);

        self.maybe_gc();

        if let Ok(mut cache) = self.buffer_cache.lock() {
            if let Some(buffer) = cache.reuse_from_cache(aligned_size) {
                self.track_allocation(aligned_size);
                return Ok(AllocatedBuffer::new(
                    buffer,
                    aligned_size,
                    lifetime,
                ));
            }
        }

        let buffer = self.allocate_from_device(aligned_size)?;
        self.track_allocation(aligned_size);

        Ok(AllocatedBuffer::new(buffer, aligned_size, lifetime))
    }
}

impl MetalAllocator for CachingAllocator {
    fn alloc_permanent(
        &self,
        size: usize,
    ) -> Result<AllocatedBuffer, AllocError> {
        self.allocate_internal(size, BufferLifetime::Permanent)
    }

    fn alloc_scratch(
        &self,
        size: usize,
    ) -> Result<AllocatedBuffer, AllocError> {
        self.allocate_internal(size, BufferLifetime::Scratch)
    }

    fn free(
        &self,
        buffer: AllocatedBuffer,
    ) {
        self.track_deallocation(buffer.size);

        let max_pool = self.max_pool_size.load(Ordering::Relaxed);
        if let Ok(mut cache) = self.buffer_cache.lock() {
            if cache.cache_size() + buffer.size <= max_pool {
                cache.recycle_to_cache(buffer.buffer, buffer.size);
                return;
            }
        }

        drop(buffer);
    }

    fn set_scratch_pool_limit(
        &self,
        size: usize,
    ) -> usize {
        self.max_pool_size.store(size, Ordering::Relaxed);
        size
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

    fn cache_memory(&self) -> usize {
        self.buffer_cache.lock().map(|c| c.cache_size()).unwrap_or(0)
    }

    fn clear_cache(&self) {
        if let Ok(mut cache) = self.buffer_cache.lock() {
            cache.clear();
        }
    }

    fn resource_options(&self) -> MTLResourceOptions {
        self.resource_options
    }
}
