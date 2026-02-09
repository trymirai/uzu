use std::{
    cell::{Cell, RefCell},
    rc::Weak,
};

use super::{
    AllocError,
    scratch_pool::{ScratchPool, align_size},
};
use crate::backends::common::{
    Backend, Buffer, BufferLifetime, Context, NativeBuffer,
};

pub struct Allocator<B: Backend> {
    context: Weak<B::Context>,
    scratch_pool: RefCell<ScratchPool<B::NativeBuffer>>,
    active_memory: Cell<usize>,
    peak_memory: Cell<usize>,
    gc_limit: Cell<usize>,
    max_pool_size: Cell<usize>,
}

impl<B: Backend> Allocator<B> {
    pub fn new(context: Weak<B::Context>) -> Self {
        Self {
            context,
            scratch_pool: RefCell::new(ScratchPool::new()),
            active_memory: Cell::new(0),
            peak_memory: Cell::new(0),
            gc_limit: Cell::new(usize::MAX),
            max_pool_size: Cell::new(usize::MAX),
        }
    }

    fn create_native_buffer(
        &self,
        size: usize,
    ) -> Result<B::NativeBuffer, AllocError> {
        let context = self.context.upgrade().ok_or_else(|| {
            AllocError::AllocationFailed {
                size,
                reason: "context has been dropped".to_string(),
            }
        })?;
        context.create_buffer(size).map_err(|e| AllocError::AllocationFailed {
            size,
            reason: e.to_string(),
        })
    }

    fn wrap_buffer(
        &self,
        native: B::NativeBuffer,
        is_scratch: bool,
    ) -> Buffer<B> {
        Buffer::new(native, self.context.clone(), is_scratch)
    }

    pub fn set_gc_limit(
        &self,
        limit: usize,
    ) {
        self.gc_limit.set(limit);
    }

    pub fn set_max_pool_size(
        &self,
        limit: usize,
    ) {
        self.max_pool_size.set(limit);
    }

    pub fn gc_limit(&self) -> usize {
        self.gc_limit.get()
    }

    pub fn max_pool_size(&self) -> usize {
        self.max_pool_size.get()
    }

    #[inline]
    fn track_allocation(
        &self,
        size: usize,
    ) {
        let new_active = self.active_memory.get() + size;
        self.active_memory.set(new_active);
        if new_active > self.peak_memory.get() {
            self.peak_memory.set(new_active);
        }
    }

    #[inline]
    fn track_deallocation(
        &self,
        size: usize,
    ) {
        self.active_memory.set(self.active_memory.get().saturating_sub(size));
    }

    fn alloc_scratch_internal(
        &self,
        aligned_size: usize,
    ) -> Result<B::NativeBuffer, AllocError> {
        let mut pool = self.scratch_pool.borrow_mut();

        if let Some((buffer, actual_size)) = pool.find_buffer(aligned_size) {
            self.track_allocation(actual_size);
            return Ok(buffer);
        }

        let gc_limit = self.gc_limit.get();
        let active = self.active_memory.get();
        let cache = pool.available_size();
        let mem_required = active + cache + aligned_size;

        if mem_required >= gc_limit {
            let to_free = mem_required.saturating_sub(gc_limit);
            pool.release_cached_buffers(to_free);
        }

        pool.total_allocated += aligned_size;
        drop(pool);

        let buffer = self.create_native_buffer(aligned_size)?;
        self.track_allocation(aligned_size);

        Ok(buffer)
    }

    pub fn alloc(
        &self,
        lifetime: BufferLifetime,
        size: usize,
    ) -> Result<Buffer<B>, AllocError> {
        let aligned_size = align_size(size);

        let (native, is_scratch) = match lifetime {
            BufferLifetime::Permanent => {
                let buffer = self.create_native_buffer(aligned_size)?;
                self.track_allocation(aligned_size);
                (buffer, false)
            },
            BufferLifetime::Scratch => {
                (self.alloc_scratch_internal(aligned_size)?, true)
            },
        };

        Ok(self.wrap_buffer(native, is_scratch))
    }

    pub fn handle_buffer_drop(
        &self,
        buffer: B::NativeBuffer,
        is_scratch: bool,
    ) {
        let size = buffer.length();
        self.track_deallocation(size);

        if is_scratch {
            let mut pool = self.scratch_pool.borrow_mut();
            pool.return_buffer(buffer, size);

            let max_pool_size = self.max_pool_size.get();
            let current_size = pool.available_size();
            if current_size > max_pool_size {
                let to_free = current_size - max_pool_size;
                pool.release_cached_buffers(to_free);
            }
        }
    }

    pub fn active_memory(&self) -> usize {
        self.active_memory.get()
    }

    pub fn peak_memory(&self) -> usize {
        self.peak_memory.get()
    }

    pub fn reset_peak_memory(&self) {
        self.peak_memory.set(self.active_memory.get());
    }

    pub fn cache_memory(&self) -> usize {
        self.scratch_pool.borrow().available_size()
    }

    pub fn clear_cache(&self) {
        self.scratch_pool.borrow_mut().clear();
    }
}
