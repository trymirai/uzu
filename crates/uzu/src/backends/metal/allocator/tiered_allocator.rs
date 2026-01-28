use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use super::super::{
    MTLBuffer, MTLDevice, MTLDeviceExt, MTLResourceOptions, ProtocolObject, Retained,
};
use super::{AllocError, AllocatedBuffer, BufferLifetime, MetalAllocator};

const PAGE_SIZE: usize = 16384;
const SMALL_BUFFER_THRESHOLD: usize = 64 * 1024;
const MIN_BUCKET_SIZE: usize = 4096;

fn next_power_of_two(size: usize) -> usize {
    if size <= MIN_BUCKET_SIZE {
        return MIN_BUCKET_SIZE;
    }
    1usize << (usize::BITS - (size - 1).leading_zeros())
}

struct SmallBufferPool {
    buckets: HashMap<usize, Vec<Retained<ProtocolObject<dyn MTLBuffer>>>>,
    total_cached: usize,
}

impl SmallBufferPool {
    fn new() -> Self {
        Self {
            buckets: HashMap::new(),
            total_cached: 0,
        }
    }

    fn find_buffer(&mut self, size: usize) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let bucket_size = next_power_of_two(size);

        if let Some(buffers) = self.buckets.get_mut(&bucket_size) {
            if let Some(buffer) = buffers.pop() {
                self.total_cached -= bucket_size;
                return Some(buffer);
            }
        }

        None
    }

    fn return_buffer(
        &mut self,
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        bucket_size: usize,
    ) {
        self.buckets.entry(bucket_size).or_default().push(buffer);
        self.total_cached += bucket_size;
    }

    fn cache_size(&self) -> usize {
        self.total_cached
    }

    fn clear(&mut self) {
        self.buckets.clear();
        self.total_cached = 0;
    }
}

struct LargeBufferPool {
    available: BTreeMap<usize, Vec<Retained<ProtocolObject<dyn MTLBuffer>>>>,
    total_cached: usize,
}

impl LargeBufferPool {
    fn new() -> Self {
        Self {
            available: BTreeMap::new(),
            total_cached: 0,
        }
    }

    fn find_buffer(&mut self, size: usize) -> Option<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)> {
        let mut found_key = None;
        for (&key, buffers) in self.available.range(size..) {
            if !buffers.is_empty() {
                found_key = Some(key);
                break;
            }
        }

        let key = found_key?;
        let buffers = self.available.get_mut(&key)?;
        let buffer = buffers.pop()?;

        if buffers.is_empty() {
            self.available.remove(&key);
        }

        self.total_cached -= key;
        Some((buffer, key))
    }

    fn return_buffer(&mut self, buffer: Retained<ProtocolObject<dyn MTLBuffer>>, size: usize) {
        self.available.entry(size).or_default().push(buffer);
        self.total_cached += size;
    }

    fn cache_size(&self) -> usize {
        self.total_cached
    }

    fn clear(&mut self) {
        self.available.clear();
        self.total_cached = 0;
    }
}

struct TieredPool {
    small: SmallBufferPool,
    large: LargeBufferPool,
}

impl TieredPool {
    fn new() -> Self {
        Self {
            small: SmallBufferPool::new(),
            large: LargeBufferPool::new(),
        }
    }

    fn cache_size(&self) -> usize {
        self.small.cache_size() + self.large.cache_size()
    }

    fn clear(&mut self) {
        self.small.clear();
        self.large.clear();
    }
}

pub struct TieredAllocator {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    pool: Mutex<TieredPool>,
    active_memory: AtomicUsize,
    peak_memory: AtomicUsize,
    resource_options: MTLResourceOptions,
}

// SAFETY: Metal objects on Apple Silicon are thread-safe.
unsafe impl Send for TieredAllocator {}
unsafe impl Sync for TieredAllocator {}

impl TieredAllocator {
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            device,
            pool: Mutex::new(TieredPool::new()),
            active_memory: AtomicUsize::new(0),
            peak_memory: AtomicUsize::new(0),
            resource_options: MTLResourceOptions::STORAGE_MODE_SHARED,
        }
    }

    pub fn with_resource_options(mut self, options: MTLResourceOptions) -> Self {
        self.resource_options = options;
        self
    }

    fn align_to_page(size: usize) -> usize {
        (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
    }

    fn is_small_buffer(size: usize) -> bool {
        size < SMALL_BUFFER_THRESHOLD
    }

    fn track_allocation(&self, size: usize) {
        let new_active = self.active_memory.fetch_add(size, Ordering::Relaxed) + size;
        self.peak_memory.fetch_max(new_active, Ordering::Relaxed);
    }

    fn track_deallocation(&self, size: usize) {
        self.active_memory.fetch_sub(size, Ordering::Relaxed);
    }

    fn allocate_from_device(
        &self,
        size: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, AllocError> {
        self.device
            .new_buffer(size, self.resource_options)
            .ok_or_else(|| AllocError::AllocationFailed {
                size,
                reason: "device.newBuffer returned nil".to_string(),
            })
    }
}

impl MetalAllocator for TieredAllocator {
    fn alloc_permanent(&self, size: usize) -> Result<AllocatedBuffer, AllocError> {
        let aligned_size = Self::align_to_page(size);
        let buffer = self.allocate_from_device(aligned_size)?;
        self.track_allocation(aligned_size);

        Ok(AllocatedBuffer::new(
            buffer,
            aligned_size,
            BufferLifetime::Permanent,
        ))
    }

    fn alloc_scratch(&self, size: usize) -> Result<AllocatedBuffer, AllocError> {
        let aligned_size = Self::align_to_page(size);

        if Self::is_small_buffer(aligned_size) {
            let bucket_size = next_power_of_two(aligned_size);

            if let Ok(mut pool) = self.pool.lock() {
                if let Some(buffer) = pool.small.find_buffer(aligned_size) {
                    self.track_allocation(bucket_size);
                    return Ok(AllocatedBuffer::new(
                        buffer,
                        bucket_size,
                        BufferLifetime::Scratch,
                    ));
                }
            }

            let buffer = self.allocate_from_device(bucket_size)?;
            self.track_allocation(bucket_size);

            Ok(AllocatedBuffer::new(
                buffer,
                bucket_size,
                BufferLifetime::Scratch,
            ))
        } else {
            if let Ok(mut pool) = self.pool.lock() {
                if let Some((buffer, actual_size)) = pool.large.find_buffer(aligned_size) {
                    self.track_allocation(actual_size);
                    return Ok(AllocatedBuffer::new(
                        buffer,
                        actual_size,
                        BufferLifetime::Scratch,
                    ));
                }
            }

            let buffer = self.allocate_from_device(aligned_size)?;
            self.track_allocation(aligned_size);

            Ok(AllocatedBuffer::new(
                buffer,
                aligned_size,
                BufferLifetime::Scratch,
            ))
        }
    }

    fn free(&self, buffer: AllocatedBuffer) {
        self.track_deallocation(buffer.size);

        match buffer.lifetime {
            BufferLifetime::Permanent => {
                drop(buffer);
            }
            BufferLifetime::Scratch => {
                if let Ok(mut pool) = self.pool.lock() {
                    if Self::is_small_buffer(buffer.size) {
                        pool.small.return_buffer(buffer.buffer, buffer.size);
                    } else {
                        pool.large.return_buffer(buffer.buffer, buffer.size);
                    }
                } else {
                    drop(buffer);
                }
            }
        }
    }

    fn active_memory(&self) -> usize {
        self.active_memory.load(Ordering::Relaxed)
    }

    fn peak_memory(&self) -> usize {
        self.peak_memory.load(Ordering::Relaxed)
    }

    fn reset_peak_memory(&self) {
        self.peak_memory
            .store(self.active_memory.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    fn cache_memory(&self) -> usize {
        self.pool.lock().map(|p| p.cache_size()).unwrap_or(0)
    }

    fn clear_cache(&self) {
        if let Ok(mut pool) = self.pool.lock() {
            pool.clear();
        }
    }

    fn resource_options(&self) -> MTLResourceOptions {
        self.resource_options
    }
}
