use std::{
    collections::HashMap,
    path::Path,
    sync::{
        Arc, Weak,
        atomic::{AtomicUsize, Ordering},
    },
};

use metal::{
    MTL4CommandBufferExt, MTL4CommandQueue, MTL4CommandQueueExt, MTLBuffer, MTLCaptureDescriptor,
    MTLCaptureDestination, MTLCaptureManager, MTLCaptureTarget, MTLComputePipelineState, MTLDevice, MTLDeviceExt,
    MTLFunctionConstantValues, MTLGPUFamily, MTLLibrary, MTLResidencySet, MTLResidencySetDescriptor,
    MTLResourceOptions, MTLSparsePageSize,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use parking_lot::{Mutex, MutexGuard};

use crate::backends::{
    common::{Allocation, AllocationPool, AllocationType, Allocator, Backend, Context, DeviceCapabilities},
    metal::{
        Metal,
        command_buffer::MetalCommandBufferInitial,
        error::MetalError,
        metal_extensions::{DeviceExt, LibraryPipelineExtensions},
        sparse::{MetalSparseBuffer, MetalSparseHeapPool, MetalSparseMappingOpsBatch},
    },
};

pub(super) const LARGE_MIN_GPU_CORES: u32 = 30;

pub struct MetalContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub gpu_core_count: u32,
    pub apple_gpu_family: MTLGPUFamily,
    pub supports_mxu: bool,
    pub device_name: String,
    pub residency_set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    pub command_queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    allocator: Arc<Allocator<Metal>>,
    peak_memory_usage: AtomicUsize,
    library_cache: Mutex<HashMap<usize, Retained<ProtocolObject<dyn MTLLibrary>>>>,
    pipeline_cache: Mutex<HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
    sparse_heap_pool: Mutex<MetalSparseHeapPool>,
    weak_self: Weak<MetalContext>,
}

impl MetalContext {
    pub(super) fn update_peak_memory_usage(&self) {
        self.peak_memory_usage.fetch_max(self.device.current_allocated_size(), Ordering::Relaxed);
    }

    fn library(
        &self,
        data: &'static [u8],
        compressed: bool,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, MetalError> {
        // `data` always comes from an `include_bytes!` constant, so its address is a stable, unique key.
        let key = data.as_ptr() as usize;
        if let Some(library) = self.library_cache.lock().get(&key) {
            return Ok(library.clone());
        }

        let maybe_uncompressed_data_owned;
        let data = if compressed {
            maybe_uncompressed_data_owned = zstd::decode_all(data).map_err(MetalError::CannotDecompressLibrary)?;

            &maybe_uncompressed_data_owned
        } else {
            data
        };

        let library = self
            .device
            .new_library_with_data(data)
            .map_err(|nserror| MetalError::CannotCreateLibrary(nserror.to_string()))?;
        self.library_cache.lock().insert(key, library.clone());

        Ok(library)
    }

    pub(super) fn compute_pipeline_state(
        &self,
        library_data: &'static [u8],
        library_compressed: bool,
        cache_key: &str,
        function_name: &str,
        constants: Option<&MTLFunctionConstantValues>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        if let Some(pipeline) = self.pipeline_cache.lock().get(cache_key) {
            return Ok(pipeline.clone());
        }

        let pipeline =
            self.library(library_data, library_compressed)?.compute_pipeline_state(function_name, constants)?;
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
        for op in mappings {
            self.command_queue.update_buffer_mappings(&op.buffer, Some(op.heap.lock().heap()), &op.mtl_operations);
        }
    }
}

impl Context for MetalContext {
    type Backend = Metal;

    fn new() -> Result<Arc<Self>, MetalError> {
        let device = <dyn MTLDevice>::system_default().ok_or(MetalError::CannotOpenDevice)?;
        let device_name = device.name();
        let gpu_core_count = device.gpu_core_count();
        let apple_gpu_family = device.newest_supported_apple_gpu_family();
        let supports_mxu = device.supports_mxu();

        let residency_set_descriptor = MTLResidencySetDescriptor::new();
        residency_set_descriptor.set_initial_capacity(1024);
        let residency_set = device
            .new_residency_set_with_descriptor(&residency_set_descriptor)
            .map_err(|nserror| MetalError::CannotCreateResidencySet(nserror.to_string()))?;

        let command_queue = device.new_mtl4_command_queue().ok_or(MetalError::CannotCreateCommandQueue)?;
        command_queue.add_residency_set(&residency_set);

        let page_size = MTLSparsePageSize::KB256;
        let heap_capacity = Metal::ALLOCATION_GRANULARITY;
        let sparse_pool = MetalSparseHeapPool::new(page_size, heap_capacity);

        Ok(Arc::new_cyclic(|weak_self| Self {
            device,
            gpu_core_count,
            apple_gpu_family,
            supports_mxu,
            device_name,
            residency_set,
            command_queue,
            allocator: Allocator::new(weak_self.clone()),
            peak_memory_usage: AtomicUsize::new(0),
            library_cache: Mutex::new(HashMap::new()),
            pipeline_cache: Mutex::new(HashMap::new()),
            sparse_heap_pool: Mutex::new(sparse_pool),
            weak_self: weak_self.clone(),
        }))
    }

    fn device_name(&self) -> Option<&str> {
        Some(&self.device_name)
    }

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        let buffer = self
            .device
            .new_buffer(size, MTLResourceOptions::STORAGE_MODE_SHARED)
            .ok_or(MetalError::CannotCreateBuffer)?;

        self.residency_set.add_allocation(buffer.as_ref());
        self.residency_set.commit();
        self.residency_set.request_residency();

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

    fn create_command_buffer(
        &self,
        name: Option<&str>,
    ) -> Result<MetalCommandBufferInitial, MetalError> {
        // TODO: reuse
        let command_allocator = self.device.new_command_allocator().ok_or(MetalError::CannotCreateCommandBuffer)?;
        let command_buffer = self.device.new_mtl4_command_buffer().ok_or(MetalError::CannotCreateCommandBuffer)?;
        command_buffer.set_label(name);
        let context = self.weak_self.upgrade().unwrap(); // never fails
        Ok(MetalCommandBufferInitial::new(command_allocator, command_buffer, context))
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
        let capture_descriptor = MTLCaptureDescriptor::new();
        capture_descriptor.set_destination(MTLCaptureDestination::GPUTraceDocument);
        capture_descriptor.set_output_path(Some(&trace_path.with_added_extension("gputrace")));
        capture_descriptor.set_capture_object(Some(&MTLCaptureTarget::Device(self.device.clone())));

        MTLCaptureManager::shared_capture_manager()
            .start_capture_with_descriptor(&capture_descriptor)
            .map_err(|error| MetalError::CannotStartGpuCapture(error.to_string()))?;

        Ok(())
    }

    fn stop_capture(&self) -> Result<(), <Self::Backend as Backend>::Error> {
        MTLCaptureManager::shared_capture_manager().stop_capture();

        Ok(())
    }

    fn device_capabilities(&self) -> DeviceCapabilities {
        let mut capabilities = DeviceCapabilities::empty();
        if self.device.supports_placement_sparse_resources() {
            capabilities |= DeviceCapabilities::SPARSE_BUFFERS;
        }
        capabilities
    }
}
