use std::{
    collections::HashMap,
    path::Path,
    sync::{
        Arc, OnceLock, Weak,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use metal::{
    MTL4CommandAllocator, MTL4CommandQueue, MTL4CommandQueueDescriptor, MTLCaptureTarget, MTLComputePipelineState,
    MTLDevice, MTLDeviceExt, MTLEvent, MTLEventExt, MTLFunctionConstantValues, MTLGPUFamily, MTLLibrary,
    MTLResidencySet, MTLResidencySetDescriptor, MTLSharedEvent, MTLSparsePageSize,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use parking_lot::Mutex;

use crate::{
    backends::{
        common::{
            Backend, BufferRef, Context, CpuBuffer, DeviceCapabilities,
            allocator::{block::BlockAllocator, pool::PoolAllocator},
            capture_matcher::CAPTURE_MATCHER,
        },
        metal::{
            Metal,
            ane::{
                AneError,
                ffn::AneFfn,
                linear::{AneLinear, AneLinearProgram, AneLinears, runs_on_ane, supported_output},
                mlp::AneMlp,
                shared::AneHealth,
            },
            buffer::{dense::MetalDenseBuffer, sparse::MetalSparseBuffer},
            capture::MetalCaptureSpan,
            command_buffer::MetalCommandBufferEncoding,
            decompression,
            error::MetalError,
            heaps::MetalHeaps,
            kernel::{BlitCopyMetalKernel, BlitFillMetalKernel},
            metal_extensions::{DeviceExt, LibraryPipelineExtensions},
        },
    },
    config::mlp::dense_mlp::DenseMLPConfig,
    encodable_block::{
        linear::{Linear, LinearRole, LinearWeights},
        mlp::{DenseMlp, DenseMlpWeights, Mlp},
    },
};

pub(super) struct SparseState {
    pub did_ops: bool,
    pub event_value: u64,
    /// Per lane, the last `event_value` its queue waited for: every queue that uses sparse
    /// buffers must itself wait for the mappings, ordering through another queue is not enough.
    pub synced: [u64; 2],
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
    lanes: [OnceLock<Arc<MetalLane>>; 2],
    ane: Mutex<AneState>,
    weak_self: Weak<MetalContext>,
}

/// A command queue with its own timeline. Prefill chunks alternate between two lanes so one chunk's
/// GPU work can overlap the other's Neural Engine work; each lane is ordered by itself, and lanes
/// are ordered against each other through their events.
pub struct MetalLane {
    pub queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    /// Shared with the ANE, which waits for its `ready` values and signals its `done` values.
    pub event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    /// Orders other lanes' GPU work after this lane's. Metal does not reliably order GPU work
    /// across queues through a shared event (the waiting queue can run ahead), a plain event does.
    pub order: Retained<ProtocolObject<dyn MTLEvent>>,
    /// Reaches an ANE round trip's `done` value once the ANE has finished it. The ANE signals
    /// `event` itself, but a queue other than the first does not reliably wait for that signal
    /// (its next command buffer runs before the ANE is done), so the completion of the request
    /// is relayed to this plain event through `relay_queue` and the queue waits for this one.
    pub done: Retained<ProtocolObject<dyn MTLEvent>>,
    relay_queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    next_value: AtomicU64,
}

impl MetalLane {
    /// Marks an ANE round trip as finished for the lane's queue (see [`Self::done`]).
    pub fn relay_done(
        &self,
        value: u64,
    ) {
        self.relay_queue.signal_event_value(&self.done, value);
    }

    /// Reserves `count` consecutive values of the lane's events, returning the first. Values are
    /// handed out in the order they are encoded, so they increase along the lane's queue.
    pub fn reserve_values(
        &self,
        count: u64,
    ) -> u64 {
        self.next_value.fetch_add(count, Ordering::AcqRel) + 1
    }
}

/// The Neural Engine executors of this device, built on demand per MLP shape and linear output width.
#[derive(Default)]
struct AneState {
    disabled: bool,
    /// Shared by every executor: the first failed request retires them all.
    health: Arc<AneHealth>,
    ffns: Vec<Arc<AneFfn>>,
    /// What every linear program shares, created with the first one.
    linears: Option<Arc<AneLinears>>,
    linear_programs: Vec<Arc<AneLinearProgram>>,
    /// ANE call sites wrapped so far (MLPs and linears); a call site's index orders its segment
    /// across pipelined prefill chunks.
    wrapped_layers: usize,
    /// Set once a dense MLP stays on the GPU: it does not order its layer across chunks, so
    /// chunks cannot be pipelined.
    has_unordered_mlps: bool,
}

impl MetalContext {
    /// The ANE executor for MLPs shaped like `weights` (built on first use), unless the ANE is
    /// disabled (`UZU_ANE=0`), unavailable, or cannot run them.
    pub fn ane_ffn(
        &self,
        weights: &DenseMlpWeights<'_, Metal>,
    ) -> Option<Arc<AneFfn>> {
        let mut ane = self.ane.lock();
        if ane.disabled {
            return None;
        }
        if let Some(ffn) = ane.ffns.iter().find(|ffn| ffn.matches(weights)) {
            return Some(ffn.clone());
        }
        match AneFfn::new(self, ane.health.clone(), weights) {
            Ok(Some(ffn)) => {
                let ffn = Arc::new(ffn);
                ane.ffns.push(ffn.clone());
                Some(ffn)
            },
            Ok(None) => None,
            Err(error) => {
                eprintln!("uzu: the Neural Engine is unavailable, MLPs run on the GPU: {error}");
                ane.disabled = true;
                None
            },
        }
    }

    /// The ANE program for linear layers shaped like `weights` (built on first use), unless the
    /// ANE is disabled (`UZU_ANE=0`), unavailable, or cannot run them.
    pub fn ane_linear(
        &self,
        weights: &LinearWeights<'_, Metal>,
    ) -> Option<Arc<AneLinearProgram>> {
        let output = supported_output(weights)?;
        let mut ane = self.ane.lock();
        if ane.disabled {
            return None;
        }
        if let Some(program) = ane.linear_programs.iter().find(|program| program.output() == output) {
            return Some(program.clone());
        }
        match self.build_linear_program(&mut ane, output) {
            Ok(program) => {
                ane.linear_programs.push(program.clone());
                Some(program)
            },
            Err(error) => {
                eprintln!("uzu: the Neural Engine is unavailable, linears run on the GPU: {error}");
                ane.disabled = true;
                None
            },
        }
    }

    fn build_linear_program(
        &self,
        ane: &mut AneState,
        output: u32,
    ) -> Result<Arc<AneLinearProgram>, AneError> {
        let shared = if let Some(shared) = &ane.linears {
            shared.clone()
        } else {
            let shared = Arc::new(AneLinears::new(self, ane.health.clone())?);
            ane.linears = Some(shared.clone());
            shared
        };
        Ok(Arc::new(AneLinearProgram::new(shared, output)?))
    }

    /// The lane with this index (0 is the main queue), created on first use.
    pub fn lane(
        &self,
        index: usize,
    ) -> Result<Arc<MetalLane>, MetalError> {
        if let Some(lane) = self.lanes[index].get() {
            return Ok(lane.clone());
        }
        let queue = if index == 0 {
            self.command_queue.clone()
        } else {
            let queue_descriptor = MTL4CommandQueueDescriptor::new();
            queue_descriptor.set_label(Some(&format!("compute queue {index}")));
            let queue = self
                .device
                .new_mtl4_command_queue_with_descriptor(&queue_descriptor)
                .map_err(|_| MetalError::CannotCreateCommandQueue)?;
            queue.add_residency_set(&self.residency_set.lock());
            queue
        };
        let event = self.device.new_shared_event().ok_or(MetalError::CannotCreateEvent)?;
        let order = self.device.new_event().ok_or(MetalError::CannotCreateEvent)?;
        let done = self.device.new_event().ok_or(MetalError::CannotCreateEvent)?;
        let relay_queue = self
            .device
            .new_mtl4_command_queue_with_descriptor(&MTL4CommandQueueDescriptor::new())
            .map_err(|_| MetalError::CannotCreateCommandQueue)?;
        let lane = Arc::new(MetalLane {
            queue,
            event,
            order,
            done,
            relay_queue,
            next_value: AtomicU64::new(0),
        });
        Ok(self.lanes[index].get_or_init(|| lane).clone())
    }

    /// Whether prefill chunks may be pipelined across lanes: a healthy ANE executor orders every
    /// MLP layer, so a chunk's layer-L GPU work waits only for the previous chunk's layer L.
    pub fn ane_pipelining_available(&self) -> bool {
        if std::env::var("UZU_ANE_PIPELINE").is_ok_and(|v| v == "0") {
            return false;
        }
        let ane = self.ane.lock();
        !ane.disabled && !ane.has_unordered_mlps && !ane.ffns.is_empty() && ane.health.is_healthy()
    }

    /// The index of the next ANE-run layer, in wrapping (model layer) order.
    pub fn next_ane_layer_index(&self) -> usize {
        let mut ane = self.ane.lock();
        ane.wrapped_layers += 1;
        ane.wrapped_layers - 1
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
            synced: [0; 2],
        };

        let block_allocator = BlockAllocator::new(16 * 1024);

        let heaps = MetalHeaps::new(device.clone(), peak_memory_usage.clone(), MTLSparsePageSize::KB256, 256);

        let context = Arc::new_cyclic(|weak_self: &Weak<Self>| Self {
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
            lanes: std::array::from_fn(|_| OnceLock::new()),
            ane: Mutex::new(AneState {
                disabled: std::env::var("UZU_ANE").is_ok_and(|value| value == "0"),
                ..Default::default()
            }),
            weak_self: weak_self.clone(),
        });
        Ok(context)
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

    fn wrap_mlp(
        &self,
        mlp: DenseMlp<Metal>,
        config: &DenseMLPConfig,
    ) -> Box<dyn Mlp<Metal>> {
        match AneMlp::wrap(self, mlp, config) {
            Ok(ane_mlp) => Box::new(ane_mlp),
            Err(dense) => {
                self.ane.lock().has_unordered_mlps = true;
                Box::new(dense)
            },
        }
    }

    fn wrap_linear(
        &self,
        linear: Box<dyn Linear<Metal>>,
        role: LinearRole,
    ) -> Box<dyn Linear<Metal>> {
        if !runs_on_ane(role) {
            return linear;
        }
        match AneLinear::wrap(self, linear, role) {
            Ok(ane_linear) => Box::new(ane_linear),
            Err(linear) => linear,
        }
    }

    fn pipelines_prefill_chunks(&self) -> bool {
        self.ane_pipelining_available()
    }
}
