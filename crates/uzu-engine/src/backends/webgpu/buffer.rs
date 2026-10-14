use std::{
    alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error},
    convert::Infallible,
    os::raw::c_void,
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use parking_lot::Mutex;

use super::{WebGpu, WebGpuError, runtime::Runtime};
use crate::{
    backends::common::{
        Buffer, BufferCpuAccessible, ConstantBuffer, GlobalBuffer, ScratchBuffer, SparseBuffer,
        allocator::{bump::BumpAllocation, pool::PoolAllocation},
    },
    utils::downcast::downcast_ref,
};

const HOST_ALIGNMENT: usize = 64;
// BufferCpuAccessible requires every accepted buffer to support a contiguous
// CPU mirror later. Rust Layout/slice sizes must fit isize even on wasm32.
const MAX_HOST_BUFFER_SIZE: usize = (isize::MAX as usize) & !(HOST_ALIGNMENT - 1);

struct HostAllocation {
    pointer: NonNull<c_void>,
    layout: Layout,
}

impl HostAllocation {
    fn new(size: usize) -> Self {
        let layout = Layout::from_size_align(size.max(4), HOST_ALIGNMENT).expect("invalid WebGPU host buffer layout");
        let pointer =
            NonNull::new(unsafe { alloc_zeroed(layout) }).unwrap_or_else(|| handle_alloc_error(layout)).cast();
        Self {
            pointer,
            layout,
        }
    }
}

// As for the CPU/Metal backends, buffer borrows and command completion govern
// access to the stable allocation. The mutex only protects lazy initialization.
unsafe impl Send for HostAllocation {}

impl Drop for HostAllocation {
    fn drop(&mut self) {
        unsafe { dealloc(self.pointer.as_ptr().cast(), self.layout) };
    }
}

pub(super) struct BufferStorage {
    pub id: u32,
    pub size: usize,
    pub runtime: Arc<Runtime>,
    host: Mutex<Option<HostAllocation>>,
    resident: AtomicBool,
    host_dirty: AtomicBool,
    gpu_dirty: AtomicBool,
}

impl BufferStorage {
    pub fn ensure_uploaded(&self) -> Result<(), WebGpuError> {
        if !self.resident.load(Ordering::Acquire) {
            self.runtime.create_buffer(self.id, self.size.next_multiple_of(4))?;
            self.resident.store(true, Ordering::Release);
        }
        if self.host_dirty.load(Ordering::Acquire) {
            let host = self.host.lock();
            let allocation = host.as_ref().expect("dirty buffer must have a host allocation");
            let bytes = unsafe { std::slice::from_raw_parts(allocation.pointer.as_ptr().cast(), self.size) };
            self.runtime.upload(self.id, bytes)?;
            self.host_dirty.store(false, Ordering::Release);
        }
        Ok(())
    }

    pub fn mark_gpu_written(&self) {
        self.gpu_dirty.store(true, Ordering::Release);
    }

    fn cpu_pointer(&self) -> NonNull<c_void> {
        let mut host = self.host.lock();
        let allocation = host.get_or_insert_with(|| {
            self.runtime.record_host_allocation(self.size);
            HostAllocation::new(self.size)
        });
        if self.gpu_dirty.load(Ordering::Acquire) {
            let bytes = unsafe { std::slice::from_raw_parts_mut(allocation.pointer.as_ptr().cast(), self.size) };
            self.runtime.read(self.id, bytes).expect("WebGPU readback failed");
            self.gpu_dirty.store(false, Ordering::Release);
        }
        // The common interface exposes one pointer for both const and mutable
        // access. Conservatively upload after either; ordinary GPU-only buffers
        // never allocate a CPU mirror or take this path.
        self.host_dirty.store(true, Ordering::Release);
        allocation.pointer
    }
}

impl Drop for BufferStorage {
    fn drop(&mut self) {
        if self.resident.load(Ordering::Relaxed) {
            self.runtime.destroy_buffer(self.id);
        }
    }
}

pub struct WebGpuBuffer {
    pub(super) storage: Arc<BufferStorage>,
}

impl WebGpuBuffer {
    pub(super) fn from_source(
        runtime: Arc<Runtime>,
        source: &wasm_bindgen::JsValue,
        offset: u64,
        size: usize,
        cancellation: &wasm_bindgen::JsValue,
    ) -> Result<Self, WebGpuError> {
        if size == 0 {
            return Err(WebGpuError::Runtime("Cannot initialize an empty weights tensor".into()));
        }
        let buffer = Self::new(runtime, size)?;
        buffer.storage.runtime.initialize_from_source(buffer.storage.id, source, offset, size, cancellation)?;
        buffer.storage.resident.store(true, Ordering::Release);
        // No CPU pointer has escaped. If a later caller needs one, it must read
        // back these weights before creating the usual stable, retained mirror.
        buffer.storage.gpu_dirty.store(true, Ordering::Release);
        Ok(buffer)
    }

    pub(super) fn new(
        runtime: Arc<Runtime>,
        size: usize,
    ) -> Result<Self, WebGpuError> {
        assert!(size > 0, "WebGPU buffer size must be nonzero");
        let limit = runtime.max_buffer_size.min(MAX_HOST_BUFFER_SIZE);
        if size > limit {
            return Err(WebGpuError::BufferTooLarge {
                requested: size,
                limit,
            });
        }
        Ok(Self {
            storage: Arc::new(BufferStorage {
                id: runtime.next_buffer_id(),
                size,
                runtime,
                host: Mutex::new(None),
                resident: AtomicBool::new(false),
                host_dirty: AtomicBool::new(false),
                gpu_dirty: AtomicBool::new(false),
            }),
        })
    }
}

impl Buffer for WebGpuBuffer {
    type Backend = WebGpu;
    fn size(&self) -> usize {
        self.storage.size
    }
}
impl BufferCpuAccessible for WebGpuBuffer {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        self.storage.cpu_pointer()
    }
}
impl GlobalBuffer for WebGpuBuffer {}

impl Buffer for BumpAllocation<WebGpuBuffer> {
    type Backend = WebGpu;
    fn size(&self) -> usize {
        self.range().iter().len()
    }
}
impl BufferCpuAccessible for BumpAllocation<WebGpuBuffer> {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        unsafe { self.page().cpu_ptr().byte_add(self.range().start) }
    }
}
impl ConstantBuffer for BumpAllocation<WebGpuBuffer> {}

impl Buffer for PoolAllocation<WebGpuBuffer> {
    type Backend = WebGpu;
    fn size(&self) -> usize {
        self.range().iter().len()
    }
}
impl ScratchBuffer for PoolAllocation<WebGpuBuffer> {}

pub struct WebGpuSparseBuffer {
    never: Infallible,
}
impl Buffer for WebGpuSparseBuffer {
    type Backend = WebGpu;
    fn size(&self) -> usize {
        match self.never {}
    }
}
impl SparseBuffer for WebGpuSparseBuffer {
    fn map(
        &mut self,
        _until: usize,
    ) -> Result<(), WebGpuError> {
        match self.never {}
    }
}

pub(super) trait WebGpuBufferExt: Buffer<Backend = WebGpu> {
    fn gpu_storage(&self) -> (Arc<BufferStorage>, usize) {
        if let Some(buffer) = downcast_ref::<WebGpuBuffer>(self) {
            (buffer.storage.clone(), 0)
        } else if let Some(buffer) = downcast_ref::<BumpAllocation<WebGpuBuffer>>(self) {
            (buffer.page().storage.clone(), buffer.range().start)
        } else if let Some(buffer) = downcast_ref::<PoolAllocation<WebGpuBuffer>>(self) {
            (buffer.page().storage.clone(), buffer.range().start)
        } else {
            unreachable!("unsupported WebGPU buffer type")
        }
    }
}
impl<T: Buffer<Backend = WebGpu> + ?Sized> WebGpuBufferExt for T {}
