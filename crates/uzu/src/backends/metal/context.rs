use std::{cell::RefCell, collections::HashMap, path::Path, rc::Rc};

use metal::{
    MTLBuffer, MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager, MTLCommandQueue, MTLCommandQueueExt,
    MTLComputePipelineState, MTLDevice, MTLDeviceExt, MTLEvent, MTLFunctionConstantValues, MTLLibrary,
    MTLResourceOptions,
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
        metal::command_buffer::MetalCommandBufferInitial,
    },
    utils::ModelSize,
};

pub struct MetalContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    allocator: Rc<Allocator<Metal>>,
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
            <dyn metal::MTLDevice>::system_default().ok_or(MetalError::CannotOpenDevice)?;

        let command_queue =
            device.new_command_queue_with_max_command_buffer_count(1024).ok_or(MetalError::CannotCreateCommandQueue)?;

        let library = device
            .new_library_with_data(kernel::MTLB)
            .map_err(|nserror| MetalError::CannotCreateLibrary(nserror.to_string()))?;

        let device_capabilities = MetalDeviceCapabilities::from_device(&device);

        Ok(Rc::new_cyclic(|weak_self| Self {
            device,
            command_queue,
            allocator: Allocator::new(weak_self.clone()),
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
        self.device.new_buffer(size, MTLResourceOptions::STORAGE_MODE_SHARED).ok_or(MetalError::CannotCreateBuffer)
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

    fn enable_capture() {
        unsafe {
            std::env::set_var("METAL_CAPTURE_ENABLED", "1");
        }
    }

    fn start_capture(
        &self,
        trace_path: &std::path::Path,
    ) -> Result<(), <Self::Backend as crate::backends::common::Backend>::Error> {
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

    fn stop_capture(&self) -> Result<(), <Self::Backend as crate::backends::common::Backend>::Error> {
        MTLCaptureManager::shared_capture_manager().stop_capture();

        Ok(())
    }
}
