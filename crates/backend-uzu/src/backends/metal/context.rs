#[cfg(test)]
use std::cell::Ref;
use std::{
    cell::{RefCell, RefMut},
    collections::HashMap,
    path::Path,
    rc::{Rc, Weak},
    sync::atomic::{AtomicU64, Ordering},
};

use metal::{
    MTL4CommandQueue, MTLBuffer, MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager, MTLCommandQueue,
    MTLCommandQueueExt, MTLComputePipelineState, MTLDevice, MTLDeviceExt, MTLEvent, MTLFunctionConstantValues,
    MTLLibrary, MTLResourceOptions, MTLSparsePageSize,
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
        common::{Allocation, AllocationPool, AllocationType, Allocator, Backend, Context},
        metal::{
            command_buffer::MetalCommandBufferInitial,
            metal_extensions::SparsePageSizeExt,
            sparse::{MetalSparseBuffer, MetalSparseHeapPool},
        },
    },
    utils::model_size::ModelSize,
};

pub struct MetalContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub command_queue4: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    timeline_event: Retained<ProtocolObject<dyn MTLEvent>>,
    timeline_value: AtomicU64,
    allocator: Rc<Allocator<Metal>>,
    peak_memory_usage: RefCell<usize>,
    device_capabilities: MetalDeviceCapabilities,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipeline_cache: RefCell<HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
    sparse_heap_pool: RefCell<MetalSparseHeapPool>,
    weak_self: Weak<MetalContext>,
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

    #[cfg(test)]
    pub(super) fn sparse_heap_pool(&self) -> Ref<'_, MetalSparseHeapPool> {
        self.sparse_heap_pool.borrow()
    }

    pub(super) fn sparse_heap_pool_mut(&self) -> RefMut<'_, MetalSparseHeapPool> {
        self.sparse_heap_pool.borrow_mut()
    }

    pub(super) fn timeline_get_and_increment(&self) -> u64 {
        self.timeline_value.fetch_add(1, Ordering::Release)
    }

    pub(super) fn timeline_event(&self) -> &ProtocolObject<dyn MTLEvent> {
        &self.timeline_event
    }
}

impl Context for MetalContext {
    type Backend = Metal;

    fn new() -> Result<Rc<Self>, MetalError> {
        let device: Retained<ProtocolObject<dyn MTLDevice>> =
            <dyn MTLDevice>::system_default().ok_or(MetalError::CannotOpenDevice)?;

        let command_queue =
            device.new_command_queue_with_max_command_buffer_count(1024).ok_or(MetalError::CannotCreateCommandQueue)?;

        let command_queue4 = device.new_mtl4_command_queue().ok_or(MetalError::CannotCreateCommandQueueMtl4)?;

        let library = device
            .new_library_with_data(kernel::MTLB)
            .map_err(|nserror| MetalError::CannotCreateLibrary(nserror.to_string()))?;

        let device_capabilities = MetalDeviceCapabilities::from_device(&device);

        let page_size = MTLSparsePageSize::KB256;
        let heap_capacity = 64 * 4 * page_size.in_bytes();
        let sparse_pool = MetalSparseHeapPool::new(page_size, heap_capacity);
        let timeline_event = device.new_event().ok_or(MetalError::CannotCreateEvent)?;

        Ok(Rc::new_cyclic(|weak_self| Self {
            device,
            command_queue,
            command_queue4,
            timeline_event,
            timeline_value: AtomicU64::new(0),
            allocator: Allocator::new(weak_self.clone()),
            peak_memory_usage: RefCell::new(0),
            device_capabilities,
            library,
            pipeline_cache: RefCell::new(HashMap::new()),
            sparse_heap_pool: RefCell::new(sparse_pool),
            weak_self: weak_self.clone(),
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
        let buffer = self
            .device
            .new_buffer(size, MTLResourceOptions::STORAGE_MODE_SHARED)
            .ok_or(MetalError::CannotCreateBuffer)?;

        let mut peak_memory_usage_borrow = self.peak_memory_usage.borrow_mut();
        *peak_memory_usage_borrow = peak_memory_usage_borrow.max(self.device.current_allocated_size());

        Ok(buffer)
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
            self.weak_self.upgrade().unwrap(), // never fails
        ))
    }

    fn create_event(&self) -> Result<Retained<ProtocolObject<dyn MTLEvent>>, MetalError> {
        self.device.new_event().ok_or(MetalError::CannotCreateEvent)
    }

    fn create_sparse_buffer(
        &self,
        capacity: usize,
    ) -> Result<<Self::Backend as Backend>::SparseBuffer, <Self::Backend as Backend>::Error> {
        let sparse_page_size = self.sparse_heap_pool.borrow().page_size();
        let context = self.weak_self.upgrade().ok_or(MetalError::CannotCreateBuffer)?;
        Ok(MetalSparseBuffer::new(context, capacity, sparse_page_size)?)
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
