use std::{
    convert::Infallible,
    path::Path,
    sync::{Arc, Weak},
};

use super::{
    WebGpu, WebGpuError,
    buffer::{WebGpuBuffer, WebGpuSparseBuffer},
    command_buffer::WebGpuCommandBufferEncoding,
    runtime::Runtime,
};
use crate::backends::common::{
    Context, DeviceCapabilities,
    allocator::{bump::BumpAllocator, pool::PoolAllocator},
};

pub struct WebGpuContext {
    pub(super) runtime: Arc<Runtime>,
    weak_self: Weak<Self>,
}

impl WebGpuContext {
    pub fn supports_subgroups(&self) -> bool {
        self.runtime.supports_subgroups
    }

    pub fn subgroup_size_range(&self) -> Option<(u32, u32)> {
        self.runtime.subgroup_size_range
    }

    pub fn subgroup_id_supported(&self) -> bool {
        self.runtime.subgroup_id_supported
    }
}

impl Context for WebGpuContext {
    type Backend = WebGpu;

    fn new() -> Result<Arc<Self>, WebGpuError> {
        let runtime = Runtime::new()?;
        Ok(Arc::new_cyclic(|weak_self| Self {
            runtime,
            weak_self: weak_self.clone(),
        }))
    }

    fn device_name(&self) -> Option<&str> {
        Some(&self.runtime.name)
    }

    fn create_command_buffer(
        &self,
        _name: Option<&str>,
        allocation_pool: Option<Arc<PoolAllocator<WebGpuBuffer>>>,
    ) -> Result<WebGpuCommandBufferEncoding, WebGpuError> {
        Ok(WebGpuCommandBufferEncoding::new(
            self.weak_self.upgrade().expect("WebGPU context is alive"),
            // Decode usually needs only one token ID. Larger constants still
            // allocate their full size; avoid copying a 64 KiB page per token.
            BumpAllocator::new(256),
            allocation_pool.unwrap_or_else(|| self.create_allocation_pool()),
        ))
    }

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<WebGpuBuffer, WebGpuError> {
        WebGpuBuffer::new(self.runtime.clone(), size)
    }

    fn create_sparse_buffer(
        &self,
        _capacity: usize,
    ) -> Result<WebGpuSparseBuffer, WebGpuError> {
        Err(WebGpuError::NotSupported)
    }

    fn create_allocation_pool(&self) -> Arc<PoolAllocator<WebGpuBuffer>> {
        PoolAllocator::new()
    }
    fn peak_memory_usage(&self) -> Option<usize> {
        None
    }
    fn start_capture(
        &self,
        _output_path: &Path,
    ) -> Result<Infallible, WebGpuError> {
        Err(WebGpuError::NotSupported)
    }
    fn device_capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities::empty()
    }
}
