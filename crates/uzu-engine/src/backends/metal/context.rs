use std::{
    collections::HashMap,
    path::Path,
    sync::{
        Arc, Weak,
        atomic::{AtomicUsize, Ordering},
    },
};

use metal::{
    MTL4CommandQueue, MTL4Compiler, MTL4CompilerDescriptor, MTLCaptureDescriptor,
    MTLCaptureDestination, MTLCaptureManager, MTLCaptureTarget,
    MTLComputePipelineState, MTLDevice, MTLDeviceExt,
    MTLFunctionConstantValues, MTLGPUFamily, MTLLibrary,
    MTLResidencySet, MTLResidencySetDescriptor, MTLSparsePageSize,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use parking_lot::Mutex;

use crate::backends::{
    common::{
        Backend, Context, DeviceCapabilities,
        allocator::{block::BlockAllocator, pool::PoolAllocator},
    },
    metal::{
        Metal,
        buffer::{dense::MetalDenseBuffer, sparse::MetalSparseBuffer},
        command_buffer::{MetalCommandBufferCache, MetalCommandBufferEncoding},
        decompression,
        error::MetalError,
        heaps::MetalHeaps,
        metal_extensions::{CompilerPipelineExtensions, DeviceExt},
    },
};

pub(super) const LARGE_MIN_GPU_CORES: u32 = 30;

pub struct MetalContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub gpu_core_count: u32,
    pub apple_gpu_family: MTLGPUFamily,
    pub supports_mxu: bool,
    pub device_name: String,
    pub(super) residency_set: Arc<Mutex<Retained<ProtocolObject<dyn MTLResidencySet>>>>,
    pub command_queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    compiler: Retained<ProtocolObject<dyn MTL4Compiler>>,
    pub(super) block_allocator: Arc<BlockAllocator<MetalDenseBuffer>>,
    pub(super) heaps: Arc<MetalHeaps>,
    pub(super) peak_memory_usage: Arc<AtomicUsize>,
    pub(super) command_buffer_cache: Mutex<Vec<MetalCommandBufferCache>>,
    library_cache: Mutex<HashMap<usize, Retained<ProtocolObject<dyn MTLLibrary>>>>,
    pipeline_cache: Mutex<HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
    weak_self: Weak<MetalContext>,
}

impl MetalContext {
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
            maybe_uncompressed_data_owned = decompression::decompress(data);
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

        let library = self.library(library_data, library_compressed)?;
        let pipeline = self.compiler.compute_pipeline_state(&library, function_name, constants)?;
        self.pipeline_cache.lock().insert(cache_key.to_string(), pipeline.clone());

        Ok(pipeline)
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

        let peak_memory_usage = Arc::new(AtomicUsize::new(0));

        let residency_set_descriptor = MTLResidencySetDescriptor::new();
        residency_set_descriptor.set_initial_capacity(1024);
        let residency_set = device
            .new_residency_set_with_descriptor(&residency_set_descriptor)
            .map_err(|nserror| MetalError::CannotCreateResidencySet(nserror.to_string()))?;

        let command_queue = device.new_mtl4_command_queue().ok_or(MetalError::CannotCreateCommandQueue)?;
        command_queue.add_residency_set(&residency_set);

        let compiler = device
            .new_compiler_with_descriptor(&MTL4CompilerDescriptor::new())
            .map_err(|error| MetalError::CannotCreateCompiler(error.to_string()))?;

        let block_allocator = BlockAllocator::new(16 * 1024);

        let heaps = MetalHeaps::new(device.clone(), peak_memory_usage.clone(), MTLSparsePageSize::KB256, 256);

        Ok(Arc::new_cyclic(|weak_self: &Weak<Self>| Self {
            device,
            gpu_core_count,
            apple_gpu_family,
            supports_mxu,
            device_name,
            residency_set: Arc::new(Mutex::new(residency_set)),
            command_queue,
            compiler,
            block_allocator,
            heaps,
            peak_memory_usage,
            command_buffer_cache: Mutex::new(Vec::with_capacity(32)),
            library_cache: Mutex::new(HashMap::new()),
            pipeline_cache: Mutex::new(HashMap::new()),
            weak_self: weak_self.clone(),
        }))
    }

    fn device_name(&self) -> Option<&str> {
        Some(&self.device_name)
    }

    fn create_command_buffer(
        &self,
        name: Option<&str>,
        allocation_pool: Option<Arc<<Metal as Backend>::AllocationPool>>,
    ) -> Result<MetalCommandBufferEncoding, MetalError> {
        MetalCommandBufferEncoding::new(self.weak_self.upgrade().unwrap(), name, allocation_pool)
    }

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<<Metal as Backend>::GlobalBuffer, MetalError> {
        self.block_allocator.allocate(size, |size| MetalDenseBuffer::new(&self.weak_self.upgrade().unwrap(), size))
    }

    fn create_sparse_buffer(
        &self,
        capacity: usize,
    ) -> Result<<Self::Backend as Backend>::SparseBuffer, <Self::Backend as Backend>::Error> {
        MetalSparseBuffer::new(self, capacity)
    }

    fn create_allocation_pool(&self) -> Arc<<Metal as Backend>::AllocationPool> {
        PoolAllocator::new()
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
