use std::{
    collections::HashMap,
    ffi::{CString, c_void},
    path::Path,
    sync::{Arc, Weak},
};

use parking_lot::Mutex;

use crate::backends::{
    amdgpu::{
        Amdgpu,
        buffer::{
            constant::{CONSTANT_PAGE_SIZE, ConstantPagePool},
            dense::{AmdgpuBuffer, BufferRecycler, peak_allocated_bytes},
            device::ScratchPageCache,
        },
        command_buffer::AmdgpuCommandBufferEncoding,
        error::AmdgpuError,
        hip::{HIP_STREAM_NON_BLOCKING, Hip, HipModule, HipStream, hip, hip_call},
        kernel::{AmdgpuFunction, shard_index},
    },
    common::{
        Backend, Context, DeviceCapabilities,
        allocator::{bump::BumpAllocator, pool::PoolAllocator},
    },
};

struct Handle(*mut c_void);

// HIP handles are thread-safe runtime objects.
unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

pub struct AmdgpuContext {
    pub(super) hip: &'static Hip,
    device_name: String,
    architecture: String,
    stream: Handle,
    /// Serializes submissions so that one command buffer's commands stay contiguous on the stream.
    pub(super) submission: Mutex<()>,
    /// Loaded code objects (with the aligned image copy they were loaded from), keyed by the address
    /// of their embedded image.
    modules: Mutex<HashMap<usize, (Handle, Vec<u64>)>>,
    functions: Mutex<HashMap<(usize, String), AmdgpuFunction>>,
    /// Free constant pages, reused across command buffers (`buffer/constant.rs`).
    constant_pages: ConstantPagePool,
    /// Global buffers the engine dropped, for reuse (`buffer/dense.rs`).
    buffer_recycler: Arc<BufferRecycler>,
    /// Free scratch pages, reused across allocation pools (`buffer/device.rs`).
    scratch_pages: ScratchPageCache,
    weak_self: Weak<AmdgpuContext>,
}

impl AmdgpuContext {
    pub(super) fn stream(&self) -> HipStream {
        self.stream.0
    }

    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    fn module(
        &self,
        image: &'static [u8],
    ) -> Result<HipModule, AmdgpuError> {
        let key = image.as_ptr() as usize;
        let mut modules = self.modules.lock();
        if let Some((module, _)) = modules.get(&key) {
            return Ok(module.0);
        }
        // the ELF loader wants an aligned image; include_bytes! only guarantees byte alignment
        let mut aligned = vec![0u64; image.len().div_ceil(8)];
        unsafe { std::ptr::copy_nonoverlapping(image.as_ptr(), aligned.as_mut_ptr().cast::<u8>(), image.len()) };
        let mut module = std::ptr::null_mut();
        hip_call!(self.hip, hipModuleLoadData(&mut module, aligned.as_ptr().cast()))?;
        modules.insert(key, (Handle(module), aligned));
        Ok(module)
    }

    /// Resolves a kernel variant by its entry name in the code objects of one source file.
    pub(super) fn constant_pages(&self) -> &ConstantPagePool {
        &self.constant_pages
    }

    pub(super) fn scratch_pages(&self) -> &ScratchPageCache {
        &self.scratch_pages
    }

    /// The copy or fill kernel of `kernel/native/copy_fill.clcpp`, if that code object compiled.
    pub(super) fn copy_fill_function(
        &self,
        which: super::command_buffer::CopyFill,
    ) -> Option<AmdgpuFunction> {
        use super::{command_buffer::CopyFill, kernel::native};
        if native::COPY_FILL.is_empty() {
            return None;
        }
        let name = match which {
            CopyFill::Copy => "uzu_copy_bytes",
            CopyFill::Fill => "uzu_fill_bytes",
            CopyFill::CopyBatch => "uzu_copy_batch",
        };
        self.function(&[native::COPY_FILL], name).ok()
    }

    pub fn function(
        &self,
        code_objects: &[&'static [u8]],
        entry_name: &str,
    ) -> Result<AmdgpuFunction, AmdgpuError> {
        if code_objects.is_empty() {
            return Err(AmdgpuError::KernelUnavailable(entry_name.into()));
        }
        let image = code_objects[shard_index(entry_name, code_objects.len())];
        if image.is_empty() {
            return Err(AmdgpuError::KernelUnavailable(entry_name.into()));
        }
        let key = (image.as_ptr() as usize, entry_name.to_string());
        if let Some(function) = self.functions.lock().get(&key) {
            return Ok(*function);
        }
        let module = self.module(image)?;
        let name = CString::new(entry_name).map_err(|_| AmdgpuError::KernelUnavailable(entry_name.into()))?;
        let mut function = std::ptr::null_mut();
        hip_call!(self.hip, hipModuleGetFunction(&mut function, module, name.as_ptr()))
            .map_err(|_| AmdgpuError::KernelUnavailable(entry_name.into()))?;
        let function = AmdgpuFunction(function);
        self.functions.lock().insert(key, function);
        Ok(function)
    }
}

/// gcnArchName from hipDeviceProp_t, found by scanning instead of hard-coding the struct layout.
fn device_architecture(
    hip: &Hip,
    device: i32,
) -> String {
    let mut properties = vec![0u8; 16 * 1024];
    if hip_call!(hip, hipGetDevicePropertiesR0600(properties.as_mut_ptr().cast(), device)).is_err() {
        return String::new();
    }
    let Some(start) = properties.windows(3).position(|w| w == b"gfx") else {
        return String::new();
    };
    let end = properties[start..].iter().position(|&b| b == 0).map_or(properties.len(), |e| start + e);
    String::from_utf8_lossy(&properties[start..end]).into_owned()
}

impl Drop for AmdgpuContext {
    fn drop(&mut self) {
        let _ = hip_call!(self.hip, hipStreamSynchronize(self.stream.0));
        self.buffer_recycler.shutdown();
        super::profile::report();
        for (_, (module, _image)) in self.modules.get_mut().drain() {
            let _ = hip_call!(self.hip, hipModuleUnload(module.0));
        }
        let _ = hip_call!(self.hip, hipStreamDestroy(self.stream.0));
    }
}

impl Context for AmdgpuContext {
    type Backend = Amdgpu;

    fn new() -> Result<Arc<Self>, AmdgpuError> {
        let hip = hip()?;
        let mut device_count = 0;
        hip_call!(hip, hipGetDeviceCount(&mut device_count))?;
        if device_count == 0 {
            return Err(AmdgpuError::NoDevice);
        }
        let device = 0;
        hip_call!(hip, hipSetDevice(device))?;

        let mut name = vec![0i8; 256];
        hip_call!(hip, hipDeviceGetName(name.as_mut_ptr().cast(), name.len() as i32, device))?;
        let device_name = unsafe { std::ffi::CStr::from_ptr(name.as_ptr().cast()) }.to_string_lossy().into_owned();
        let architecture = device_architecture(hip, device);

        let mut stream = std::ptr::null_mut();
        hip_call!(hip, hipStreamCreateWithFlags(&mut stream, HIP_STREAM_NON_BLOCKING))?;

        Ok(Arc::new_cyclic(|weak_self| AmdgpuContext {
            hip,
            device_name,
            architecture,
            stream: Handle(stream),
            submission: Mutex::new(()),
            modules: Mutex::new(HashMap::new()),
            functions: Mutex::new(HashMap::new()),
            constant_pages: ConstantPagePool::default(),
            buffer_recycler: BufferRecycler::new(stream),
            scratch_pages: ScratchPageCache::default(),
            weak_self: weak_self.clone(),
        }))
    }

    fn device_name(&self) -> Option<&str> {
        Some(&self.device_name)
    }

    fn create_command_buffer(
        &self,
        name: Option<&str>,
        allocation_pool: Option<Arc<<Amdgpu as Backend>::AllocationPool>>,
    ) -> Result<AmdgpuCommandBufferEncoding, AmdgpuError> {
        let constant_allocator = BumpAllocator::new(CONSTANT_PAGE_SIZE);
        let allocation_pool = allocation_pool.unwrap_or_else(|| self.create_allocation_pool());
        let mut encoding =
            AmdgpuCommandBufferEncoding::new(constant_allocator, allocation_pool, self.weak_self.upgrade().unwrap());
        if super::profile::timeline() {
            encoding.set_name(name.unwrap_or("-"));
        }
        Ok(encoding)
    }

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<<Amdgpu as Backend>::GlobalBuffer, AmdgpuError> {
        AmdgpuBuffer::recycled(size, &self.buffer_recycler)
    }

    fn create_sparse_buffer(
        &self,
        _capacity: usize,
    ) -> Result<<Amdgpu as Backend>::SparseBuffer, AmdgpuError> {
        Err(AmdgpuError::NotSupported)
    }

    fn create_allocation_pool(&self) -> Arc<<Amdgpu as Backend>::AllocationPool> {
        PoolAllocator::new()
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        Some(peak_allocated_bytes())
    }

    fn enable_capture() {}

    fn start_capture(
        &self,
        _trace_path: &Path,
    ) -> Result<(), AmdgpuError> {
        Err(AmdgpuError::NotSupported)
    }

    fn stop_capture(&self) -> Result<(), AmdgpuError> {
        Err(AmdgpuError::NotSupported)
    }

    fn device_capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities::empty()
    }
}
