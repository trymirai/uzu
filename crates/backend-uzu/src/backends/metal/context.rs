use std::{cell::RefCell, collections::HashMap, path::Path, rc::Rc};

use backend_uzu::backends::common::Backend;
use metal::{
    MTL4CommandQueue, MTLBuffer, MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager, MTLCommandQueue,
    MTLCommandQueueExt, MTLComputePipelineState, MTLDevice, MTLDeviceExt, MTLEvent, MTLFunctionConstantValues,
    MTLGPUFamily, MTLLibrary, MTLResourceOptions,
};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::{
    Metal,
    device_capabilities::MetalDeviceCapabilities,
    error::MetalError,
    kernel,
    metal_extensions::{DeviceGeneration, LibraryPipelineExtensions},
};
use crate::{
    backends::{
        common::{Allocation, AllocationPool, AllocationType, Allocator, Context},
        metal::{command_buffer::MetalCommandBufferInitial, sparse_buffer::MetalSparseBuffer},
    },
    utils::model_size::ModelSize,
};

pub struct MetalContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub command_queue4: Option<Retained<ProtocolObject<dyn MTL4CommandQueue>>>,
    allocator: Rc<Allocator<Metal>>,
    peak_memory_usage: RefCell<usize>,
    device_capabilities: MetalDeviceCapabilities,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipeline_cache: RefCell<HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
}

impl MetalContext {
    pub fn device_generation(&self) -> DeviceGeneration {
        self.device_capabilities.generation
    }

    pub fn compute_pipeline_state(
        &self,
        cache_key: &str,
        function_name: &str,
        constants: Option<&MTLFunctionConstantValues>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        if let Some(pipeline) = self.pipeline_cache.borrow().get(cache_key) {
            return Ok(pipeline.clone());
        }

        let pipeline = self.library.compute_pipeline_state(function_name, constants)?;

        self.pipeline_cache.borrow_mut().insert(cache_key.to_string(), pipeline.clone());

        Ok(pipeline)
    }
}

impl Context for MetalContext {
    type Backend = Metal;

    fn new() -> Result<Rc<Self>, MetalError> {
        let device: Retained<ProtocolObject<dyn MTLDevice>> =
            <dyn MTLDevice>::system_default().ok_or(MetalError::CannotOpenDevice)?;

        let command_queue =
            device.new_command_queue_with_max_command_buffer_count(1024).ok_or(MetalError::CannotCreateCommandQueue)?;

        let command_queue4 = if device.supports_family(MTLGPUFamily::Metal4) {
            Some(device.new_mtl4_command_queue().ok_or(MetalError::CannotCreateCommandQueueMtl4)?)
        } else {
            None
        };

        let library = device
            .new_library_with_data(kernel::MTLB)
            .map_err(|nserror| MetalError::CannotCreateLibrary(nserror.to_string()))?;

        let device_capabilities = MetalDeviceCapabilities::from_device(&device);

        Ok(Rc::new_cyclic(|weak_self| Self {
            device,
            command_queue,
            command_queue4,
            allocator: Allocator::new(weak_self.clone()),
            peak_memory_usage: RefCell::new(0),
            device_capabilities,
            library,
            pipeline_cache: RefCell::new(HashMap::new()),
        }))
    }

    fn is_high_performance(&self) -> bool {
        self.device_capabilities.is_high_performance()
    }

    fn recommended_async_batch_size(
        &self,
        model_path: &Path,
    ) -> usize {
        let cores = self.device_capabilities.gpu_core_count;
        let model_size = ModelSize::from_path(model_path);
        match (model_size, cores) {
            (ModelSize::Large, c) if c > 20 => 32,
            (ModelSize::Large, c) if c > 10 => 16,
            (ModelSize::Large, _) => 8,
            (ModelSize::Small, c) if c > 20 => 256,
            (ModelSize::Small, c) if c > 10 => 128,
            (ModelSize::Small, _) => 64,
        }
    }

    fn debug_active(&self) -> bool {
        let upper = std::env::var("METAL_DEVICE_WRAPPER_TYPE").unwrap_or_default().to_ascii_uppercase();
        matches!(upper.as_str(), "1" | "YES" | "TRUE")
    }

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        let buffer =
            self.device.new_buffer(size, MTLResourceOptions::STORAGE_MODE_SHARED).ok_or(MetalError::CannotCreateBuffer);
        let mut peak_memory_usage_borrow = self.peak_memory_usage.borrow_mut();
        *peak_memory_usage_borrow = peak_memory_usage_borrow.max(self.device.current_allocated_size());
        buffer
    }

    fn create_allocation(
        &self,
        size: usize,
        allocation_type: AllocationType<Metal>,
    ) -> Result<Allocation<Metal>, MetalError> {
        self.allocator.allocate(size, allocation_type)
    }

    fn create_allocation_pool(
        &self,
        reusable: bool,
    ) -> AllocationPool<Metal> {
        self.allocator.create_pool(reusable)
    }

    fn create_command_buffer(&self) -> Result<MetalCommandBufferInitial, MetalError> {
        Ok(MetalCommandBufferInitial::new(
            self.command_queue.command_buffer().ok_or(MetalError::CannotCreateCommandBuffer)?,
        ))
    }

    fn create_event(&self) -> Result<Retained<ProtocolObject<dyn MTLEvent>>, MetalError> {
        self.device.new_event().ok_or(MetalError::CannotCreateEvent)
    }

    fn create_sparse_buffer(
        &self,
        capacity: usize,
    ) -> Result<<Self::Backend as Backend>::SparseBuffer, <Self::Backend as Backend>::Error> {
        if self.command_queue4.is_none() {
            return Err(MetalError::SparseQueueNotAvailable);
        }
        Ok(MetalSparseBuffer::new(self, capacity)?)
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        Some(*self.peak_memory_usage.borrow())
    }

    fn enable_capture() {
        unsafe {
            std::env::set_var("METAL_CAPTURE_ENABLED", "1");
        }
    }

    fn start_capture(
        &self,
        trace_path: &Path,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        let capture_manager = MTLCaptureManager::shared_capture_manager();
        let capture_descriptor = MTLCaptureDescriptor::new();
        capture_descriptor.set_destination(MTLCaptureDestination::GPUTraceDocument);
        capture_descriptor.set_output_path(Some(trace_path));

        self.command_queue.set_label(Some("uzu_command_queue"));
        capture_descriptor.set_capture_object(Some(self.command_queue.as_ref()));

        capture_manager
            .start_capture_with_descriptor_error(&capture_descriptor)
            .map_err(|nserror| MetalError::CannotStartGpuCapture(nserror.to_string()))?;

        Ok(())
    }

    fn stop_capture(&self) -> Result<(), <Self::Backend as Backend>::Error> {
        MTLCaptureManager::shared_capture_manager().stop_capture();

        Ok(())
    }
}
