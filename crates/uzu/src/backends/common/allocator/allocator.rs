use std::{
    collections::HashSet,
    sync::{
        Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use super::{
    AllocError, BufferLifetime,
    scratch_pool::{PAGE_SIZE, ScratchPool, next_power_of_two},
};
use crate::backends::common::Backend;

pub struct Allocator<B: Backend> {
    device: B::Device,
    resource_options: B::ResourceOptions,
    scratch_pool: Mutex<ScratchPool<B::Buffer>>,
    scratch_buffers: Mutex<HashSet<usize>>,
    active_memory: AtomicUsize,
    peak_memory: AtomicUsize,
    allocation_count: AtomicU64,
    eviction_threshold: AtomicU64,
}

impl<B: Backend> Allocator<B> {
    pub fn new(
        device: B::Device,
        resource_options: B::ResourceOptions,
    ) -> Self {
        Self {
            device,
            resource_options,
            scratch_pool: Mutex::new(ScratchPool::new()),
            scratch_buffers: Mutex::new(HashSet::new()),
            active_memory: AtomicUsize::new(0),
            peak_memory: AtomicUsize::new(0),
            allocation_count: AtomicU64::new(0),
            eviction_threshold: AtomicU64::new(1000),
        }
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

    fn track_scratch_buffer(
        &self,
        buffer: &B::Buffer,
    ) {
        use super::Buffer;
        let id = buffer.id();
        if let Ok(mut set) = self.scratch_buffers.lock() {
            set.insert(id);
        }
    }

    fn alloc_scratch_internal(
        &self,
        aligned_size: usize,
    ) -> Result<B::Buffer, AllocError> {
        use crate::backends::common::Device;

        if let Ok(mut pool) = self.scratch_pool.lock() {
            pool.tick();

            if ScratchPool::<B::Buffer>::is_small(aligned_size) {
                let bucket_size = next_power_of_two(aligned_size);

                if let Some(buffer) = pool.find_small_buffer(aligned_size) {
                    self.track_allocation(bucket_size);
                    self.track_scratch_buffer(&buffer);
                    return Ok(buffer);
                }

                drop(pool);

                let buffer = self
                    .device
                    .create_buffer(bucket_size, self.resource_options)?;
                self.track_allocation(bucket_size);
                self.track_scratch_buffer(&buffer);

                if let Ok(mut pool) = self.scratch_pool.lock() {
                    pool.total_allocated += bucket_size;
                }

                return Ok(buffer);
            } else {
                if let Some((buffer, actual_size)) =
                    pool.find_large_buffer(aligned_size)
                {
                    self.track_allocation(actual_size);
                    self.track_scratch_buffer(&buffer);
                    return Ok(buffer);
                }

                drop(pool);

                let buffer = self
                    .device
                    .create_buffer(aligned_size, self.resource_options)?;
                self.track_allocation(aligned_size);
                self.track_scratch_buffer(&buffer);

                if let Ok(mut pool) = self.scratch_pool.lock() {
                    pool.total_allocated += aligned_size;
                }

                return Ok(buffer);
            }
        }

        let alloc_size = if ScratchPool::<B::Buffer>::is_small(aligned_size) {
            next_power_of_two(aligned_size)
        } else {
            aligned_size
        };

        let buffer =
            self.device.create_buffer(alloc_size, self.resource_options)?;
        self.track_allocation(alloc_size);
        self.track_scratch_buffer(&buffer);

        Ok(buffer)
    }

    pub fn alloc(
        &self,
        lifetime: BufferLifetime,
        size: usize,
    ) -> Result<B::Buffer, AllocError> {
        use crate::backends::common::Device;

        let aligned_size = Self::align_to_page(size);

        match lifetime {
            BufferLifetime::Permanent => {
                let buffer = self
                    .device
                    .create_buffer(aligned_size, self.resource_options)?;
                self.track_allocation(aligned_size);
                Ok(buffer)
            },
            BufferLifetime::Scratch => {
                self.alloc_scratch_internal(aligned_size)
            },
        }
    }

    pub fn free(
        &self,
        buffer: B::Buffer,
    ) {
        use super::Buffer;

        let id = buffer.id();
        let size = buffer.length();

        self.track_deallocation(size);

        let is_scratch = self
            .scratch_buffers
            .lock()
            .map(|mut set| set.remove(&id))
            .unwrap_or(false);

        if is_scratch {
            if let Ok(mut pool) = self.scratch_pool.lock() {
                if ScratchPool::<B::Buffer>::is_small(size) {
                    pool.return_small_buffer(buffer, size);
                } else {
                    pool.return_large_buffer(buffer, size);
                }
            }
        }
    }

    pub fn active_memory(&self) -> usize {
        self.active_memory.load(Ordering::Relaxed)
    }

    pub fn peak_memory(&self) -> usize {
        self.peak_memory.load(Ordering::Relaxed)
    }

    pub fn reset_peak_memory(&self) {
        self.peak_memory.store(
            self.active_memory.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }

    pub fn cache_memory(&self) -> usize {
        self.scratch_pool.lock().map(|p| p.available_size()).unwrap_or(0)
    }

    pub fn clear_cache(&self) {
        if let Ok(mut pool) = self.scratch_pool.lock() {
            pool.clear();
        }
        if let Ok(mut set) = self.scratch_buffers.lock() {
            set.clear();
        }
    }
}
