use std::{
    collections::HashMap,
    path::Path,
    sync::{
        Arc, Weak,
        atomic::{AtomicUsize, Ordering},
    },
};

#[cfg(test)]
use metal::MTLSharedEvent;
use metal::{
    MTL4CommandQueue, MTL4CommandQueueExt, MTLBuffer, MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager,
    MTLCommandBuffer, MTLCommandBufferExt, MTLCommandQueue, MTLCommandQueueExt, MTLComputePipelineState, MTLDevice,
    MTLDeviceExt, MTLEvent, MTLFunctionConstantValues, MTLLibrary, MTLResourceOptions, MTLSparsePageSize,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use parking_lot::{Mutex, MutexGuard};

use super::{
    Metal,
    device_profile::{DeviceProfile, classify_device},
    error::MetalError,
    metal_extensions::{DeviceExt, LibraryPipelineExtensions},
};
use crate::backends::{
    common::{Allocation, AllocationPool, AllocationType, Allocator, Backend, Context, DeviceCapabilities},
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
    /// Cross-queue ticket allocation and submission ordering.
    timeline: Mutex<TimelineState>,
    allocator: Arc<Allocator<Metal>>,
    peak_memory_usage: AtomicUsize,
    library_cache: Mutex<HashMap<usize, Retained<ProtocolObject<dyn MTLLibrary>>>>,
    pipeline_cache: Mutex<HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
    sparse_heap_pool: Mutex<MetalSparseHeapPool>,
    device_profile: DeviceProfile,
    weak_self: Weak<MetalContext>,
    #[cfg(test)]
    timeline_shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
}

#[derive(Debug, Default)]
struct TimelineState {
    scheduled_value: u64,
    mapping_signal_value: u64,
    compute_waited_mapping_value: u64,
}

impl TimelineState {
    /// Reserve one mapping operation after every previously scheduled queue operation.
    fn reserve_mapping(&mut self) -> (u64, u64) {
        let wait_value = self.scheduled_value;
        let signal_value = wait_value + 1;
        self.scheduled_value = signal_value;
        self.mapping_signal_value = signal_value;
        (wait_value, signal_value)
    }

    /// Reserve one compute submit and report the newest mapping it must wait for.
    fn reserve_compute(&mut self) -> (Option<u64>, u64) {
        let mapping_wait =
            (self.mapping_signal_value > self.compute_waited_mapping_value).then_some(self.mapping_signal_value);
        if let Some(mapping_wait) = mapping_wait {
            self.compute_waited_mapping_value = mapping_wait;
        }

        let signal_value = self.scheduled_value + 1;
        self.scheduled_value = signal_value;
        (mapping_wait, signal_value)
    }
}

impl MetalContext {
    pub fn supports_mxu(&self) -> bool {
        self.device.supports_mxu()
    }

    pub fn device_profile(&self) -> DeviceProfile {
        self.device_profile
    }

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

    pub fn compute_pipeline_state(
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
        if mappings.is_empty() {
            return;
        }

        // Hold the submission lock until the complete MTL4 batch is queued.
        // A compute submit cannot reserve a later ticket and commit first.
        let mut timeline = self.timeline.lock();
        let (wait_value, signal_value) = timeline.reserve_mapping();
        self.command_queue4.wait_for_event_value(&self.timeline_event, wait_value);
        for op in mappings {
            self.command_queue4.update_buffer_mappings(&op.buffer, Some(op.heap.lock().heap()), &op.mtl_operations);
        }
        self.command_queue4.signal_event_value(&self.timeline_event, signal_value);
        drop(timeline);

        // This line prevent tests from freezing, showing pink screen and shutting down computer
        #[cfg(test)]
        self.timeline_shared_event.wait_until_signaled_value_timeout_ms(wait_value, 10);
    }

    /// Commit a compute command buffer in the same order as its timeline ticket.
    pub(super) fn submit_compute(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) {
        // The short lock covers ticket allocation and both commits. In
        // particular, a mapping wait is not marked consumed until the wait
        // buffer is on the compute queue immediately ahead of this work.
        let mut timeline = self.timeline.lock();
        let (mapping_wait, signal_value) = timeline.reserve_compute();
        if let Some(mapping_wait) = mapping_wait {
            let wait_buffer = self.command_queue.command_buffer().expect("Failed to create sparse mapping wait");
            wait_buffer.set_label(Some("sync (sparse mapping wait)"));
            wait_buffer.encode_wait_for_event_value(&self.timeline_event, mapping_wait);
            wait_buffer.commit();
        }

        command_buffer.encode_signal_event_value(&self.timeline_event, signal_value);
        command_buffer.commit();
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

        let gpu_core_count = device.gpu_core_count();
        let device_profile = classify_device(
            gpu_core_count,
            device.supports_family(metal::MTLGPUFamily::Apple8),
            device.supports_family(metal::MTLGPUFamily::Apple9),
            device.supports_mxu(),
        );
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
            timeline: Mutex::new(TimelineState::default()),
            allocator: Allocator::new(weak_self.clone()),
            peak_memory_usage: AtomicUsize::new(0),
            library_cache: Mutex::new(HashMap::new()),
            pipeline_cache: Mutex::new(HashMap::new()),
            sparse_heap_pool: Mutex::new(sparse_pool),
            device_profile,
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

    fn create_command_buffer(
        &self,
        name: Option<&str>,
    ) -> Result<MetalCommandBufferInitial, MetalError> {
        let command_buffer = self.command_queue.command_buffer().ok_or(MetalError::CannotCreateCommandBuffer)?;
        command_buffer.set_label(name);
        let context = self.weak_self.upgrade().unwrap(); // never fails
        Ok(MetalCommandBufferInitial::new(command_buffer, context))
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
        capture_descriptor.set_output_path(Some(&trace_path.with_added_extension("gputrace")));

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

    fn device_capabilities(&self) -> DeviceCapabilities {
        let mut capabilities = DeviceCapabilities::empty();
        if self.device.supports_placement_sparse_resources() {
            capabilities |= DeviceCapabilities::SPARSE_BUFFERS;
        }
        capabilities
    }
}

#[cfg(test)]
mod timeline_tests {
    use proc_macros::uzu_test;

    use super::{MetalContext, TimelineState};
    use crate::{
        backends::common::{Context, Encoder, SparseBuffer},
        tests::helpers::{
            alloc_allocation, alloc_allocation_with_data, allocation_to_vec, sparse_buffer_create,
        },
    };

    #[uzu_test]
    fn mapping_wait_is_consumed_only_by_the_next_reserved_compute_submit() {
        let mut timeline = TimelineState::default();

        assert_eq!(timeline.reserve_compute(), (None, 1));
        assert_eq!(timeline.reserve_mapping(), (1, 2));
        assert_eq!(timeline.reserve_compute(), (Some(2), 3));
        assert_eq!(timeline.reserve_compute(), (None, 4));
    }

    #[uzu_test]
    fn interleaved_operations_form_one_monotonic_cross_queue_chain() {
        let mut timeline = TimelineState::default();

        assert_eq!(timeline.reserve_mapping(), (0, 1));
        assert_eq!(timeline.reserve_mapping(), (1, 2));
        assert_eq!(timeline.reserve_compute(), (Some(2), 3));
        assert_eq!(timeline.reserve_mapping(), (3, 4));
        assert_eq!(timeline.reserve_compute(), (Some(4), 5));
    }

    #[uzu_test]
    fn compute_mapping_compute_chain_exposes_the_new_sparse_pages() {
        const BYTE_COUNT: usize = 4096;

        let context = MetalContext::new().expect("create Metal context");
        let mut first_marker = alloc_allocation::<crate::backends::metal::Metal, u8>(&context, BYTE_COUNT);
        let mut first_encoder = Encoder::new(context.as_ref()).expect("create first encoder");
        first_encoder.encode_fill(&mut first_marker, 0xa5);
        let first_pending = first_encoder.end_encoding().submit();

        let mut sparse = sparse_buffer_create::<crate::backends::metal::Metal>(&context, BYTE_COUNT);
        sparse.map(&context, &(0..1)).expect("map sparse page between compute submits");

        let expected: Vec<u8> = (0..BYTE_COUNT).map(|index| (index % 251) as u8).collect();
        let source = alloc_allocation_with_data::<crate::backends::metal::Metal, u8>(&context, &expected);
        let mut readback = alloc_allocation::<crate::backends::metal::Metal, u8>(&context, BYTE_COUNT);
        let mut marker_readback = alloc_allocation::<crate::backends::metal::Metal, u8>(&context, BYTE_COUNT);
        let mut second_encoder = Encoder::new(context.as_ref()).expect("create second encoder");
        second_encoder.encode_copy(&source, .., &mut sparse, ..BYTE_COUNT);
        second_encoder.encode_copy(&sparse, ..BYTE_COUNT, &mut readback, ..);
        second_encoder.encode_copy(&first_marker, .., &mut marker_readback, ..);
        let second_pending = second_encoder.end_encoding().submit();

        second_pending.wait_until_completed().expect("complete compute after mapping");
        first_pending.wait_until_completed().expect("complete compute before mapping");
        assert_eq!(allocation_to_vec::<crate::backends::metal::Metal, u8>(&readback), expected);
        assert!(
            allocation_to_vec::<crate::backends::metal::Metal, u8>(&marker_readback)
                .into_iter()
                .all(|byte| byte == 0xa5),
            "second compute observed data before the first compute completed",
        );
    }
}
