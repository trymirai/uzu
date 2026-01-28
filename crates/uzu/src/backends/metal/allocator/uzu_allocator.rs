use std::{
    collections::BTreeMap,
    sync::{
        Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use super::{
    super::{
        MTLBuffer, MTLDevice, MTLDeviceExt, MTLResourceOptions, ProtocolObject,
        Retained,
    },
    AllocError, AllocatedBuffer, BufferLifetime, MetalAllocator,
};

const PAGE_SIZE: usize = 16384;
const SMALL_BUFFER_THRESHOLD: usize = 64 * 1024;
const SIZE_TOLERANCE: f64 = 0.1;

fn next_power_of_two(size: usize) -> usize {
    if size <= 4096 {
        return 4096;
    }
    1usize << (usize::BITS - (size - 1).leading_zeros())
}

struct CachedBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    last_used_tick: u64,
}

struct ScratchPool {
    small_buckets: BTreeMap<usize, Vec<CachedBuffer>>,
    large_exact: BTreeMap<usize, Vec<CachedBuffer>>,
    total_allocated: usize,
    limit: usize,
    current_tick: u64,
}

impl ScratchPool {
    fn new() -> Self {
        Self {
            small_buckets: BTreeMap::new(),
            large_exact: BTreeMap::new(),
            total_allocated: 0,
            limit: usize::MAX,
            current_tick: 0,
        }
    }

    fn is_small(size: usize) -> bool {
        size < SMALL_BUFFER_THRESHOLD
    }

    fn tick(&mut self) -> u64 {
        self.current_tick += 1;
        self.current_tick
    }

    fn find_small_buffer(
        &mut self,
        size: usize,
    ) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let bucket_size = next_power_of_two(size);

        let buffers = self.small_buckets.get_mut(&bucket_size)?;
        let cached = buffers.pop()?;

        if buffers.is_empty() {
            self.small_buckets.remove(&bucket_size);
        }

        Some(cached.buffer)
    }

    fn find_large_buffer(
        &mut self,
        size: usize,
    ) -> Option<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)> {
        let max_size = ((size as f64) * (1.0 + SIZE_TOLERANCE)).ceil() as usize;

        let mut found_key = None;
        for (&key, buffers) in self.large_exact.range(size..=max_size) {
            if !buffers.is_empty() {
                found_key = Some(key);
                break;
            }
        }

        let key = found_key?;
        let buffers = self.large_exact.get_mut(&key)?;
        let cached = buffers.pop()?;

        if buffers.is_empty() {
            self.large_exact.remove(&key);
        }

        Some((cached.buffer, key))
    }

    fn return_small_buffer(
        &mut self,
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        bucket_size: usize,
    ) {
        let tick = self.current_tick;
        self.small_buckets.entry(bucket_size).or_default().push(CachedBuffer {
            buffer,
            last_used_tick: tick,
        });
    }

    fn return_large_buffer(
        &mut self,
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        size: usize,
    ) {
        let tick = self.current_tick;
        self.large_exact.entry(size).or_default().push(CachedBuffer {
            buffer,
            last_used_tick: tick,
        });
    }

    fn evict_stale(
        &mut self,
        max_age: u64,
    ) -> usize {
        let current = self.current_tick;
        let threshold = current.saturating_sub(max_age);

        let mut evicted = 0;

        let keys_to_check: Vec<_> =
            self.small_buckets.keys().copied().collect();
        for key in keys_to_check {
            if let Some(buffers) = self.small_buckets.get_mut(&key) {
                let before = buffers.len();
                buffers.retain(|b| b.last_used_tick >= threshold);
                evicted += before - buffers.len();
                if buffers.is_empty() {
                    self.small_buckets.remove(&key);
                }
            }
        }

        let keys_to_check: Vec<_> = self.large_exact.keys().copied().collect();
        for key in keys_to_check {
            if let Some(buffers) = self.large_exact.get_mut(&key) {
                let before = buffers.len();
                buffers.retain(|b| b.last_used_tick >= threshold);
                evicted += before - buffers.len();
                if buffers.is_empty() {
                    self.large_exact.remove(&key);
                }
            }
        }

        evicted
    }

    fn available_size(&self) -> usize {
        let small: usize = self
            .small_buckets
            .iter()
            .map(|(size, buffers)| size * buffers.len())
            .sum();

        let large: usize = self
            .large_exact
            .iter()
            .map(|(size, buffers)| size * buffers.len())
            .sum();

        small + large
    }

    fn clear(&mut self) -> usize {
        let small_count: usize =
            self.small_buckets.values().map(|v| v.len()).sum();
        let large_count: usize =
            self.large_exact.values().map(|v| v.len()).sum();
        let freed = self.available_size();

        self.small_buckets.clear();
        self.large_exact.clear();
        self.total_allocated -= freed;

        small_count + large_count
    }
}

pub struct UzuAllocator {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    scratch_pool: Mutex<ScratchPool>,
    active_memory: AtomicUsize,
    peak_memory: AtomicUsize,
    allocation_count: AtomicU64,
    eviction_threshold: AtomicU64,
    resource_options: MTLResourceOptions,
}

// SAFETY: Metal objects on Apple Silicon are thread-safe.
// The device has its own internal synchronization.
// The scratch_pool is protected by a Mutex.
unsafe impl Send for UzuAllocator {}
unsafe impl Sync for UzuAllocator {}

impl UzuAllocator {
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            device,
            scratch_pool: Mutex::new(ScratchPool::new()),
            active_memory: AtomicUsize::new(0),
            peak_memory: AtomicUsize::new(0),
            allocation_count: AtomicU64::new(0),
            eviction_threshold: AtomicU64::new(1000),
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

    pub fn set_eviction_threshold(
        &self,
        max_age: u64,
    ) {
        self.eviction_threshold.store(max_age, Ordering::Relaxed);
    }

    pub fn evict_stale_buffers(&self) -> usize {
        let threshold = self.eviction_threshold.load(Ordering::Relaxed);
        if let Ok(mut pool) = self.scratch_pool.lock() {
            pool.evict_stale(threshold)
        } else {
            0
        }
    }

    fn align_to_page(size: usize) -> usize {
        (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
    }

    fn track_allocation(
        &self,
        size: usize,
    ) {
        let new_active =
            self.active_memory.fetch_add(size, Ordering::Relaxed) + size;
        self.peak_memory.fetch_max(new_active, Ordering::Relaxed);
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
    }

    fn track_deallocation(
        &self,
        size: usize,
    ) {
        self.active_memory.fetch_sub(size, Ordering::Relaxed);
    }

    fn allocate_from_device(
        &self,
        size: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, AllocError> {
        self.device.new_buffer(size, self.resource_options).ok_or_else(|| {
            AllocError::AllocationFailed {
                size,
                reason: "device.newBuffer returned nil".to_string(),
            }
        })
    }
}

impl MetalAllocator for UzuAllocator {
    fn alloc_permanent(
        &self,
        size: usize,
    ) -> Result<AllocatedBuffer, AllocError> {
        let aligned_size = Self::align_to_page(size);
        let buffer = self.allocate_from_device(aligned_size)?;
        self.track_allocation(aligned_size);

        Ok(AllocatedBuffer::new(
            buffer,
            aligned_size,
            BufferLifetime::Permanent,
        ))
    }

    fn alloc_scratch(
        &self,
        size: usize,
    ) -> Result<AllocatedBuffer, AllocError> {
        let aligned_size = Self::align_to_page(size);

        if let Ok(mut pool) = self.scratch_pool.lock() {
            pool.tick();

            if ScratchPool::is_small(aligned_size) {
                let bucket_size = next_power_of_two(aligned_size);

                if let Some(buffer) = pool.find_small_buffer(aligned_size) {
                    self.track_allocation(bucket_size);
                    return Ok(AllocatedBuffer::new(
                        buffer,
                        bucket_size,
                        BufferLifetime::Scratch,
                    ));
                }

                drop(pool);

                let buffer = self.allocate_from_device(bucket_size)?;
                self.track_allocation(bucket_size);

                if let Ok(mut pool) = self.scratch_pool.lock() {
                    pool.total_allocated += bucket_size;
                }

                return Ok(AllocatedBuffer::new(
                    buffer,
                    bucket_size,
                    BufferLifetime::Scratch,
                ));
            } else {
                if let Some((buffer, actual_size)) =
                    pool.find_large_buffer(aligned_size)
                {
                    self.track_allocation(actual_size);
                    return Ok(AllocatedBuffer::new(
                        buffer,
                        actual_size,
                        BufferLifetime::Scratch,
                    ));
                }

                drop(pool);

                let buffer = self.allocate_from_device(aligned_size)?;
                self.track_allocation(aligned_size);

                if let Ok(mut pool) = self.scratch_pool.lock() {
                    pool.total_allocated += aligned_size;
                }

                return Ok(AllocatedBuffer::new(
                    buffer,
                    aligned_size,
                    BufferLifetime::Scratch,
                ));
            }
        }

        let alloc_size = if ScratchPool::is_small(aligned_size) {
            next_power_of_two(aligned_size)
        } else {
            aligned_size
        };

        let buffer = self.allocate_from_device(alloc_size)?;
        self.track_allocation(alloc_size);

        Ok(AllocatedBuffer::new(buffer, alloc_size, BufferLifetime::Scratch))
    }

    fn free(
        &self,
        buffer: AllocatedBuffer,
    ) {
        self.track_deallocation(buffer.size);

        match buffer.lifetime {
            BufferLifetime::Permanent => {
                drop(buffer);
            },
            BufferLifetime::Scratch => {
                if let Ok(mut pool) = self.scratch_pool.lock() {
                    if ScratchPool::is_small(buffer.size) {
                        pool.return_small_buffer(buffer.buffer, buffer.size);
                    } else {
                        pool.return_large_buffer(buffer.buffer, buffer.size);
                    }
                } else {
                    drop(buffer);
                }
            },
        }
    }

    fn set_scratch_pool_limit(
        &self,
        size: usize,
    ) -> usize {
        if let Ok(mut pool) = self.scratch_pool.lock() {
            pool.limit = size;
        }
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
        self.scratch_pool.lock().map(|p| p.available_size()).unwrap_or(0)
    }

    fn clear_cache(&self) {
        if let Ok(mut pool) = self.scratch_pool.lock() {
            pool.clear();
        }
    }

    fn resource_options(&self) -> MTLResourceOptions {
        self.resource_options
    }
}
