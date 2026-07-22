use std::{
    collections::HashMap,
    path::Path,
    sync::{
        Arc, Weak,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

#[cfg(test)]
use metal::MTLSharedEvent;
use metal::{
    MTL4CommandQueue, MTL4CommandQueueExt, MTLBuffer, MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager,
    MTLCommandQueue, MTLCommandQueueExt, MTLComputePipelineState, MTLDevice, MTLDeviceExt, MTLEvent,
    MTLFunctionConstantValues, MTLLibrary, MTLResourceOptions, MTLSparsePageSize,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use parking_lot::{Mutex, MutexGuard};

use super::{
    Metal,
    device_tier::{DeviceTier, device_tier_for_device},
    error::MetalError,
    kernel,
    metal_extensions::{DeviceExt, LibraryPipelineExtensions},
};
use crate::backends::{
    common::{Allocation, AllocationPool, AllocationType, Allocator, Backend, Context},
    metal::{
        command_buffer::MetalCommandBufferInitial,
        sparse::{MetalSparseBuffer, MetalSparseHeapPool, MetalSparseMappingOpsBatch},
    },
};

pub struct MetalContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub command_queue4: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    timeline_event: Retained<ProtocolObject<dyn MTLEvent>>,
    timeline_value: AtomicU64,
    allocator: Arc<Allocator<Metal>>,
    peak_memory_usage: AtomicUsize,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipeline_cache: Mutex<HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
    sparse_heap_pool: Mutex<MetalSparseHeapPool>,
    device_tier: DeviceTier,
    weak_self: Weak<MetalContext>,
    #[cfg(test)]
    timeline_shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
}

impl MetalContext {
    pub fn supports_mxu(&self) -> bool {
        self.device.supports_mxu()
    }

    pub(crate) fn device_tier(&self) -> DeviceTier {
        self.device_tier
    }

    pub(super) fn update_peak_memory_usage(&self) {
        self.peak_memory_usage.fetch_max(self.device.current_allocated_size(), Ordering::Relaxed);
    }

    pub fn compute_pipeline_state(
        &self,
        cache_key: &str,
        function_name: &str,
        constants: Option<&MTLFunctionConstantValues>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        if let Some(pipeline) = self.pipeline_cache.lock().get(cache_key) {
            return Ok(pipeline.clone());
        }

        let pipeline = self.library.compute_pipeline_state(function_name, constants)?;
        self.pipeline_cache.lock().insert(cache_key.to_string(), pipeline.clone());

        Ok(pipeline)
    }

    pub(super) fn sparse_heap_pool(&self) -> MutexGuard<'_, MetalSparseHeapPool> {
        self.sparse_heap_pool.lock()
    }

    pub(super) fn sparse_update_mappings(
        &self,
        mappings: &[MetalSparseMappingOpsBatch],
    ) {
        if mappings.is_empty() {
            return;
        }

        let wait_value = self.timeline_get_and_increment();
        self.command_queue4.wait_for_event_value(&self.timeline_event, wait_value);
        for op in mappings {
            self.command_queue4.update_buffer_mappings(&op.buffer, Some(op.heap.lock().heap()), &op.mtl_operations);
        }
        self.command_queue4.signal_event_value(&self.timeline_event, wait_value + 1);

        // This line prevent tests from freezing, showing pink screen and shutting down computer
        #[cfg(test)]
        self.timeline_shared_event.wait_until_signaled_value_timeout_ms(wait_value, 10);
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

    fn new() -> Result<Arc<Self>, MetalError> {
        let device: Retained<ProtocolObject<dyn MTLDevice>> =
            <dyn MTLDevice>::system_default().ok_or(MetalError::CannotOpenDevice)?;

        let command_queue =
            device.new_command_queue_with_max_command_buffer_count(1024).ok_or(MetalError::CannotCreateCommandQueue)?;

        let command_queue4 = device.new_mtl4_command_queue().ok_or(MetalError::CannotCreateCommandQueueMtl4)?;

        let library = device
            .new_library_with_data(kernel::MTLB)
            .map_err(|nserror| MetalError::CannotCreateLibrary(nserror.to_string()))?;

        let gpu_core_count = device.gpu_core_count();
        let device_tier = device_tier_for_device(gpu_core_count, device.as_ref());
        let page_size = MTLSparsePageSize::KB256;
        let heap_capacity = Metal::ALLOCATION_GRANULARITY;
        let sparse_pool = MetalSparseHeapPool::new(page_size, heap_capacity);
        let timeline_event = device.new_event().ok_or(MetalError::CannotCreateEvent)?;
        #[cfg(test)]
        let timeline_shared_event = device.new_shared_event().ok_or(MetalError::CannotCreateEvent)?;

        Ok(Arc::new_cyclic(|weak_self| Self {
            device,
            command_queue,
            command_queue4,
            timeline_event,
            timeline_value: AtomicU64::new(0),
            allocator: Allocator::new(weak_self.clone()),
            peak_memory_usage: AtomicUsize::new(0),
            library,
            pipeline_cache: Mutex::new(HashMap::new()),
            sparse_heap_pool: Mutex::new(sparse_pool),
            device_tier,
            weak_self: weak_self.clone(),
            #[cfg(test)]
            timeline_shared_event,
        }))
    }

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        let buffer = self
            .device
            .new_buffer(size, MTLResourceOptions::STORAGE_MODE_SHARED)
            .ok_or(MetalError::CannotCreateBuffer)?;

        self.update_peak_memory_usage();

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

    fn create_sparse_buffer(
        &self,
        capacity: usize,
    ) -> Result<<Self::Backend as Backend>::SparseBuffer, <Self::Backend as Backend>::Error> {
        let sparse_page_size = self.sparse_heap_pool.lock().page_size();
        let context = self.weak_self.upgrade().ok_or(MetalError::CannotCreateBuffer)?;
        MetalSparseBuffer::new(context, capacity, sparse_page_size)
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        Some(self.peak_memory_usage.load(Ordering::Relaxed))
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

    fn sparse_buffers_supported(&self) -> bool {
        self.device.supports_placement_sparse_resources()
    }

    fn supports_symmetric_int8_activations(&self) -> bool {
        self.device.supports_mxu()
    }
}
