use std::{
    collections::HashMap,
    os::raw::c_void,
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use parking_lot::Mutex;

use crate::backends::{
    amdgpu::{
        Amdgpu,
        error::AmdgpuError,
        hip::{
            HIP_EVENT_DISABLE_TIMING, HIP_HOST_MALLOC_DEFAULT, HIP_HOST_MALLOC_NON_COHERENT, HipEvent, HipStream, hip,
            hip_call,
        },
    },
    common::{Buffer, BufferCpuAccessible, GlobalBuffer},
};

static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

pub(in crate::backends::amdgpu) fn peak_allocated_bytes() -> usize {
    PEAK_ALLOCATED_BYTES.load(Ordering::Relaxed)
}

pub(super) fn track_allocation(size: usize) {
    let allocated = ALLOCATED_BYTES.fetch_add(size, Ordering::Relaxed) + size;
    PEAK_ALLOCATED_BYTES.fetch_max(allocated, Ordering::Relaxed);
}

pub(super) fn track_release(size: usize) {
    ALLOCATED_BYTES.fetch_sub(size, Ordering::Relaxed);
}

/// Pinned allocations are coarse-grained (`hipHostMallocNonCoherent`): the GPU may cache them in L2, and CPU and GPU
/// see each other's writes at command boundaries, which is how the engine shares buffers (the CPU touches them
/// between command buffers). The Windows default is fine-grained, which the GPU does not cache: every access
/// goes to the fabric, and kernels gathering small values (quantization scales and zero points) ran 3.4x
/// slower. `UZU_AMDGPU_HOST_COHERENT=1` restores fine-grained buffers.
fn host_malloc_flags() -> std::os::raw::c_uint {
    static FLAGS: std::sync::OnceLock<std::os::raw::c_uint> = std::sync::OnceLock::new();
    *FLAGS.get_or_init(|| {
        if std::env::var("UZU_AMDGPU_HOST_COHERENT").is_ok_and(|value| value != "0") {
            HIP_HOST_MALLOC_DEFAULT
        } else {
            HIP_HOST_MALLOC_NON_COHERENT
        }
    })
}

/// Buffers up to this size are kept for reuse when the engine drops them.
const RECYCLED_BUFFER_MAX: usize = 16 << 20;
/// Bound on the memory kept for reuse.
const RECYCLED_BYTES_MAX: usize = 256 << 20;

/// Pinned buffers the engine dropped, kept for reuse by size. The engine creates small buffers per step
/// (sampling output, tree readbacks) and drops them after reading; hipHostMalloc takes milliseconds and
/// hipHostFree waits for all queued GPU work, which stalled the speculation loop (~1000 frees per generation).
/// A dropped buffer comes back with an event recorded on the context's stream and is handed out again once
/// that event has completed, i.e. once every command issued before the drop has finished with it.
pub(in crate::backends::amdgpu) struct BufferRecycler {
    stream: usize,
    alive: AtomicBool,
    released: Mutex<HashMap<usize, Vec<(usize, usize)>>>,
    released_bytes: AtomicUsize,
}

impl std::fmt::Debug for BufferRecycler {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        formatter.debug_struct("BufferRecycler").finish_non_exhaustive()
    }
}

impl BufferRecycler {
    pub(in crate::backends::amdgpu) fn new(stream: HipStream) -> Arc<Self> {
        Arc::new(Self {
            stream: stream as usize,
            alive: AtomicBool::new(true),
            released: Mutex::new(HashMap::new()),
            released_bytes: AtomicUsize::new(0),
        })
    }

    /// A released buffer of `allocation_size` the GPU is done with.
    fn take(
        &self,
        allocation_size: usize,
    ) -> Option<NonNull<c_void>> {
        let hip = hip().ok()?;
        let mut released = self.released.lock();
        let entries = released.get_mut(&allocation_size)?;
        let index = entries.iter().position(|&(_, event)| hip_call!(hip, hipEventQuery(event as HipEvent)).is_ok())?;
        let (ptr, event) = entries.swap_remove(index);
        let _ = hip_call!(hip, hipEventDestroy(event as HipEvent));
        self.released_bytes.fetch_sub(allocation_size, Ordering::Relaxed);
        NonNull::new(ptr as *mut c_void)
    }

    /// Keeps the buffer for reuse after the commands issued so far; false if it should be freed instead.
    fn release(
        &self,
        ptr: NonNull<c_void>,
        allocation_size: usize,
    ) -> bool {
        if !self.alive.load(Ordering::Acquire)
            || allocation_size > RECYCLED_BUFFER_MAX
            || self.released_bytes.load(Ordering::Relaxed) + allocation_size > RECYCLED_BYTES_MAX
        {
            return false;
        }
        let Ok(hip) = hip() else {
            return false;
        };
        let mut event = std::ptr::null_mut();
        if hip_call!(hip, hipEventCreateWithFlags(&mut event, HIP_EVENT_DISABLE_TIMING)).is_err() {
            return false;
        }
        if hip_call!(hip, hipEventRecord(event, self.stream as HipStream)).is_err() {
            let _ = hip_call!(hip, hipEventDestroy(event));
            return false;
        }
        self.released.lock().entry(allocation_size).or_default().push((ptr.as_ptr() as usize, event as usize));
        self.released_bytes.fetch_add(allocation_size, Ordering::Relaxed);
        true
    }

    /// Frees everything kept; later releases free directly. Called before the stream goes away.
    pub(in crate::backends::amdgpu) fn shutdown(&self) {
        self.alive.store(false, Ordering::Release);
        let Ok(hip) = hip() else {
            return;
        };
        for (allocation_size, entries) in self.released.lock().drain() {
            for (ptr, event) in entries {
                let _ = hip_call!(hip, hipEventSynchronize(event as HipEvent));
                let _ = hip_call!(hip, hipEventDestroy(event as HipEvent));
                let _ = hip_call!(hip, hipHostFree(ptr as *mut c_void));
                track_release(allocation_size);
            }
        }
    }
}

/// Host-mapped (pinned) memory. On an APU it is the same physical memory the GPU reads, and HIP's
/// unified addressing makes the host pointer valid on the device.
#[derive(Debug)]
pub struct AmdgpuBuffer {
    ptr: NonNull<c_void>,
    size: usize,
    recycler: Option<Arc<BufferRecycler>>,
}

impl AmdgpuBuffer {
    /// A buffer that goes back to `recycler` when dropped, taken from it when one of the size is free.
    pub(in crate::backends::amdgpu) fn recycled(
        size: usize,
        recycler: &Arc<BufferRecycler>,
    ) -> Result<Self, AmdgpuError> {
        let allocation_size = size.max(64);
        if let Some(ptr) = recycler.take(allocation_size) {
            // as fresh allocations: zero-filled
            unsafe { ptr.as_ptr().cast::<u8>().write_bytes(0, allocation_size) };
            return Ok(Self {
                ptr,
                size,
                recycler: Some(recycler.clone()),
            });
        }
        let mut buffer = Self::new(size)?;
        buffer.recycler = Some(recycler.clone());
        Ok(buffer)
    }

    pub(in crate::backends::amdgpu) fn new(size: usize) -> Result<Self, AmdgpuError> {
        let hip = hip()?;
        let allocation_size = size.max(64);
        let mut ptr = std::ptr::null_mut();
        let started = std::time::Instant::now();
        hip_call!(hip, hipHostMalloc(&mut ptr, allocation_size, host_malloc_flags()))?;
        crate::backends::amdgpu::profile::record_memory_call("hipHostMalloc", started.elapsed());
        let ptr = NonNull::new(ptr).ok_or(AmdgpuError::Hip {
            call: "hipHostMalloc",
            code: 0,
            message: "returned a null pointer".into(),
        })?;
        // Metal hands out zero-filled buffers and the engine may rely on it; pinned host memory is recycled.
        unsafe { ptr.as_ptr().cast::<u8>().write_bytes(0, allocation_size) };
        track_allocation(allocation_size);
        Ok(Self {
            ptr,
            size,
            recycler: None,
        })
    }

    pub(in crate::backends::amdgpu) fn device_address(&self) -> u64 {
        self.ptr.as_ptr() as u64
    }
}

// The allocation has a stable address; callers synchronize access through buffer borrows and command buffers.
unsafe impl Send for AmdgpuBuffer {}
unsafe impl Sync for AmdgpuBuffer {}

impl Drop for AmdgpuBuffer {
    fn drop(&mut self) {
        if let Some(recycler) = &self.recycler
            && recycler.release(self.ptr, self.size.max(64))
        {
            return;
        }
        if let Ok(hip) = hip() {
            let started = std::time::Instant::now();
            let _ = hip_call!(hip, hipHostFree(self.ptr.as_ptr()));
            crate::backends::amdgpu::profile::record_memory_call("hipHostFree", started.elapsed());
        }
        track_release(self.size.max(64));
    }
}

impl Buffer for AmdgpuBuffer {
    type Backend = Amdgpu;

    fn size(&self) -> usize {
        self.size
    }
}

impl BufferCpuAccessible for AmdgpuBuffer {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        self.ptr
    }
}

impl GlobalBuffer for AmdgpuBuffer {}
