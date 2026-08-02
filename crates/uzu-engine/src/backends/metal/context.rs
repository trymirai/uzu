use std::{
    collections::HashMap,
    path::Path,
    sync::{
        Arc, OnceLock, Weak,
        atomic::{AtomicUsize, Ordering},
    },
};

use metal::{
    MTL4CommandAllocator, MTL4CommandQueue, MTL4CommandQueueDescriptor, MTLCaptureTarget, MTLComputePipelineState,
    MTLDevice, MTLDeviceExt, MTLEvent, MTLEventExt, MTLFunctionConstantValues, MTLGPUFamily, MTLLibrary,
    MTLResidencySet, MTLResidencySetDescriptor, MTLSparsePageSize,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use parking_lot::Mutex;

use crate::backends::{
    common::{
        Backend, BufferRef, Context, CpuBuffer, DeviceCapabilities,
        allocator::{block::BlockAllocator, pool::PoolAllocator},
        capture_matcher::CAPTURE_MATCHER,
    },
    metal::{
        Metal,
        buffer::{dense::MetalDenseBuffer, sparse::MetalSparseBuffer},
        capture::MetalCaptureSpan,
        command_buffer::MetalCommandBufferEncoding,
        decompression,
        error::MetalError,
        heaps::MetalHeaps,
        kernel::{BlitCopyMetalKernel, BlitFillMetalKernel},
        metal_extensions::{DeviceExt, LibraryPipelineExtensions},
    },
};

pub(super) const LARGE_MIN_GPU_CORES: u32 = 30;

pub(super) struct SparseState {
    pub did_ops: bool,
    pub event_value: u64,
}

pub struct MetalContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub gpu_core_count: u32,
    pub apple_gpu_family: MTLGPUFamily,
    pub supports_mxu: bool,
    pub device_name: String,
    pub(super) residency_set: Arc<Mutex<Retained<ProtocolObject<dyn MTLResidencySet>>>>,
    pub command_queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    pub(super) sparse_queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    pub(super) sparse_event: Retained<ProtocolObject<dyn MTLEvent>>,
    pub(super) sparse_state: Mutex<SparseState>,
    pub(super) block_allocator: Arc<BlockAllocator<MetalDenseBuffer>>,
    pub(super) heaps: Arc<MetalHeaps>,
    pub(super) peak_memory_usage: Arc<AtomicUsize>,
    pub(super) command_allocator_cache: Mutex<Vec<Retained<ProtocolObject<dyn MTL4CommandAllocator>>>>,
    library_cache: Mutex<HashMap<usize, Retained<ProtocolObject<dyn MTLLibrary>>>>,
    pipeline_cache: Mutex<HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
    pub(super) blit_copy: [OnceLock<BlitCopyMetalKernel>; 3],
    pub(super) blit_fill: [OnceLock<BlitFillMetalKernel>; 2],
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
        let pipeline = library.compute_pipeline_state(function_name, constants, Some(cache_key))?;
        self.pipeline_cache.lock().insert(cache_key.to_string(), pipeline.clone());

        Ok(pipeline)
    }
}

impl Context for MetalContext {
    type Backend = Metal;

    async fn new() -> Result<Arc<Self>, MetalError> {
        if CAPTURE_MATCHER.enabled() {
            unsafe {
                std::env::set_var("METAL_CAPTURE_ENABLED", "1");
            }
        }

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

        let queue_descriptor = MTL4CommandQueueDescriptor::new();
        queue_descriptor.set_label(Some("compute queue"));
        let command_queue = device
            .new_mtl4_command_queue_with_descriptor(&queue_descriptor)
            .map_err(|_| MetalError::CannotCreateCommandQueue)?;
        command_queue.add_residency_set(&residency_set);

        queue_descriptor.set_label(Some("sparse queue"));
        let sparse_queue = device
            .new_mtl4_command_queue_with_descriptor(&queue_descriptor)
            .map_err(|_| MetalError::CannotCreateCommandQueue)?;

        let sparse_event = device.new_event().ok_or(MetalError::CannotCreateEvent)?;
        sparse_event.set_label(Some("sparse event"));

        let sparse_state = SparseState {
            did_ops: false,
            event_value: 0,
        };

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
            sparse_queue,
            sparse_event,
            sparse_state: Mutex::new(sparse_state),
            block_allocator,
            heaps,
            peak_memory_usage,
            command_allocator_cache: Mutex::new(Vec::with_capacity(32)),
            library_cache: Mutex::new(HashMap::new()),
            pipeline_cache: Mutex::new(HashMap::new()),
            blit_copy: std::array::from_fn(|_| OnceLock::new()),
            blit_fill: std::array::from_fn(|_| OnceLock::new()),
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

    fn create_buffer_gpu(
        &self,
        size: usize,
    ) -> Result<<Metal as Backend>::GlobalGpuBuffer, MetalError> {
        self.block_allocator.allocate(size, |size| MetalDenseBuffer::new(&self.weak_self.upgrade().unwrap(), size))
    }

    fn create_buffer_cpu_gpu(
        &self,
        size: usize,
    ) -> Result<<Metal as Backend>::GlobalBufferCpuGpu, MetalError> {
        self.block_allocator.allocate(size, |size| MetalDenseBuffer::new(&self.weak_self.upgrade().unwrap(), size))
    }

    fn create_sparse_buffer(
        &self,
        capacity: usize,
    ) -> Result<<Self::Backend as Backend>::SparseBuffer, <Self::Backend as Backend>::Error> {
        MetalSparseBuffer::new(self.weak_self.upgrade().unwrap(), capacity)
    }

    async fn copyout<T: bytemuck::AnyBitPattern>(
        &self,
        buffer: impl BufferRef<Backend = Metal, Buffer = <Self::Backend as Backend>::GlobalGpuBuffer>,
    ) -> Result<Vec<T>, MetalError> {
        let (buffer, range) = buffer.parts();
        let bytes = unsafe {
            std::slice::from_raw_parts(buffer.cpu_ptr().as_ptr().cast::<u8>().add(range.start), range.iter().len())
        };
        Ok(bytemuck::cast_slice(bytes).to_vec())
    }

    fn create_allocation_pool(&self) -> Arc<<Metal as Backend>::AllocationPool> {
        PoolAllocator::new()
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        Some(self.peak_memory_usage.load(Ordering::Relaxed))
    }

    fn start_capture(
        &self,
        output_path: &Path,
    ) -> Result<MetalCaptureSpan, MetalError> {
        MetalCaptureSpan::new(&MTLCaptureTarget::Device(self.device.clone()), output_path)
    }

    fn device_capabilities(&self) -> DeviceCapabilities {
        let mut capabilities = DeviceCapabilities::GLOBAL_BUFFERS_CPU_ACCESSIBLE;
        if self.device.supports_placement_sparse_resources() {
            capabilities |= DeviceCapabilities::SPARSE_BUFFERS;
        }
        capabilities
    }
}
