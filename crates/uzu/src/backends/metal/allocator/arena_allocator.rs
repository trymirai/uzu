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
    AllocError, AllocatedBuffer, BufferLifetime, MetalAllocator,
};

const PAGE_SIZE: usize = 16384;

struct ArenaState {
    heap: Retained<ProtocolObject<dyn MTLHeap>>,
    heap_size: usize,
    current_offset: usize,
    buffers_in_use: usize,
}

pub struct ArenaAllocator {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    arena: Mutex<Option<ArenaState>>,
    active_memory: AtomicUsize,
    peak_memory: AtomicUsize,
    resource_options: MTLResourceOptions,
}

// SAFETY: Metal objects on Apple Silicon are thread-safe.
// The device and heap have their own internal synchronization.
unsafe impl Send for ArenaAllocator {}
unsafe impl Sync for ArenaAllocator {}

impl ArenaAllocator {
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            device,
            arena: Mutex::new(None),
            active_memory: AtomicUsize::new(0),
            peak_memory: AtomicUsize::new(0),
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

    pub fn init_arena(
        &self,
        max_scratch_size: usize,
    ) -> Result<(), AllocError> {
        let aligned_size = Self::align_to_page(max_scratch_size);

        let descriptor = MTLHeapDescriptor::new();
        descriptor.set_size(aligned_size);
        descriptor.set_storage_mode(MTLStorageMode::Shared);

        let heap = self
            .device
            .new_heap_with_descriptor(&descriptor)
            .ok_or_else(|| AllocError::AllocationFailed {
                size: aligned_size,
                reason: "Failed to create scratch heap".to_string(),
            })?;

        let mut arena_guard =
            self.arena.lock().map_err(|_| AllocError::AllocationFailed {
                size: aligned_size,
                reason: "Failed to acquire arena lock".to_string(),
            })?;

        *arena_guard = Some(ArenaState {
            heap,
            heap_size: aligned_size,
            current_offset: 0,
            buffers_in_use: 0,
        });

        Ok(())
    }

    pub fn reset_arena(&self) {
        if let Ok(mut arena_guard) = self.arena.lock() {
            if let Some(ref mut state) = *arena_guard {
                state.current_offset = 0;
                state.buffers_in_use = 0;
            }
        }
        self.active_memory.store(0, Ordering::Relaxed);
    }

    pub fn arena_usage(&self) -> (usize, usize) {
        self.arena
            .lock()
            .ok()
            .and_then(|guard| {
                guard
                    .as_ref()
                    .map(|state| (state.current_offset, state.heap_size))
            })
            .unwrap_or((0, 0))
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

    fn allocate_from_arena(
        &self,
        size: usize,
    ) -> Result<Option<Retained<ProtocolObject<dyn MTLBuffer>>>, AllocError>
    {
        let mut arena_guard =
            self.arena.lock().map_err(|_| AllocError::AllocationFailed {
                size,
                reason: "Failed to acquire arena lock".to_string(),
            })?;

        let state = match arena_guard.as_mut() {
            Some(s) => s,
            None => return Ok(None),
        };

        let aligned_size = Self::align_to_page(size);

        if state.current_offset + aligned_size > state.heap_size {
            return Ok(None);
        }

        let buffer = state
            .heap
            .new_buffer_with_offset(
                state.current_offset,
                self.resource_options,
                aligned_size,
            )
            .ok_or_else(|| AllocError::AllocationFailed {
                size: aligned_size,
                reason: "heap.newBufferWithOffset returned nil".to_string(),
            })?;

        state.current_offset += aligned_size;
        state.buffers_in_use += 1;

        Ok(Some(buffer))
    }
}

impl MetalAllocator for ArenaAllocator {
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

        if let Some(buffer) = self.allocate_from_arena(aligned_size)? {
            self.track_allocation(aligned_size);
            return Ok(AllocatedBuffer::new(
                buffer,
                aligned_size,
                BufferLifetime::Scratch,
            ));
        }

        let buffer = self.allocate_from_device(aligned_size)?;
        self.track_allocation(aligned_size);

        Ok(AllocatedBuffer::new(buffer, aligned_size, BufferLifetime::Scratch))
    }

    fn free(
        &self,
        buffer: AllocatedBuffer,
    ) {
        self.track_deallocation(buffer.size);
        drop(buffer);
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
        let (used, total) = self.arena_usage();
        total - used
    }

    fn clear_cache(&self) {
        self.reset_arena();
    }

    fn resource_options(&self) -> MTLResourceOptions {
        self.resource_options
    }
}
