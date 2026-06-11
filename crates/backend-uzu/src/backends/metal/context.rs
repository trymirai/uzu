use std::{
    cell::{Ref, RefCell, RefMut},
    collections::HashMap,
    path::Path,
    rc::{Rc, Weak},
    sync::atomic::{AtomicU64, Ordering},
};

#[cfg(test)]
use metal::MTLSharedEvent;
use metal::{
    MTL4CommandQueue, MTL4CommandQueueExt, MTLBuffer, MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager,
    MTLCommandQueue, MTLCommandQueueExt, MTLComputePipelineState, MTLDevice, MTLDeviceExt, MTLEvent,
    MTLFunctionConstantValues, MTLLibrary, MTLResourceOptions, MTLSparsePageSize,
};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::{
    Metal,
    error::MetalError,
    kernel,
    metal_extensions::{DeviceExt, LibraryPipelineExtensions},
};
use crate::{
    backends::{
        common::{Allocation, AllocationPool, AllocationType, Allocator, Backend, Context},
        metal::{
            command_buffer::MetalCommandBufferInitial,
            sparse::{MetalSparseBuffer, MetalSparseHeapPool, MetalSparseMappingOpsBatch},
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
    gpu_core_count: u32,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipeline_cache: RefCell<HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
    sparse_heap_pool: RefCell<MetalSparseHeapPool>,
    weak_self: Weak<MetalContext>,
    #[cfg(test)]
    timeline_shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
}

impl MetalContext {
    pub fn supports_mxu(&self) -> bool {
        self.device.supports_mxu()
    }

    pub fn gpu_core_count(&self) -> u32 {
        self.gpu_core_count
    }

    /// False on the M1/A14 (G13) generation, true on M2/A15 and newer.
    pub fn supports_apple8_family(&self) -> bool {
        self.device.supports_family(metal::MTLGPUFamily::Apple8)
    }

    /// False on the M2/A15/A16 generation and older, true on M3/A17 and newer.
    pub fn supports_apple9_family(&self) -> bool {
        self.device.supports_family(metal::MTLGPUFamily::Apple9)
    }

    pub(super) fn update_peak_memory_usage(&self) {
        let mut peak_memory_usage_borrow = self.peak_memory_usage.borrow_mut();
        *peak_memory_usage_borrow = peak_memory_usage_borrow.max(self.device.current_allocated_size());
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

    pub(super) fn sparse_heap_pool(&self) -> Ref<'_, MetalSparseHeapPool> {
        self.sparse_heap_pool.borrow()
    }

    pub(super) fn sparse_heap_pool_mut(&self) -> RefMut<'_, MetalSparseHeapPool> {
        self.sparse_heap_pool.borrow_mut()
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
            self.command_queue4.update_buffer_mappings(&op.buffer, Some(op.heap.borrow().heap()), &op.mtl_operations);
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

    fn new() -> Result<Rc<Self>, MetalError> {
        let device: Retained<ProtocolObject<dyn MTLDevice>> =
            <dyn MTLDevice>::system_default().ok_or(MetalError::CannotOpenDevice)?;

        let command_queue =
            device.new_command_queue_with_max_command_buffer_count(1024).ok_or(MetalError::CannotCreateCommandQueue)?;

        let command_queue4 = device.new_mtl4_command_queue().ok_or(MetalError::CannotCreateCommandQueueMtl4)?;

        let library = device
            .new_library_with_data(kernel::MTLB)
            .map_err(|nserror| MetalError::CannotCreateLibrary(nserror.to_string()))?;

        let gpu_core_count = device.gpu_core_count();

        let page_size = MTLSparsePageSize::KB256;
        let heap_capacity = Metal::ALLOCATION_GRANULARITY;
        let sparse_pool = MetalSparseHeapPool::new(page_size, heap_capacity);
        let timeline_event = device.new_event().ok_or(MetalError::CannotCreateEvent)?;
        #[cfg(test)]
        let timeline_shared_event = device.new_shared_event().ok_or(MetalError::CannotCreateEvent)?;

        Ok(Rc::new_cyclic(|weak_self| Self {
            device,
            command_queue,
            command_queue4,
            timeline_event,
            timeline_value: AtomicU64::new(0),
            allocator: Allocator::new(weak_self.clone()),
            peak_memory_usage: RefCell::new(0),
            gpu_core_count,
            library,
            pipeline_cache: RefCell::new(HashMap::new()),
            sparse_heap_pool: RefCell::new(sparse_pool),
            weak_self: weak_self.clone(),
            #[cfg(test)]
            timeline_shared_event,
        }))
    }

    fn recommended_async_batch_size(
        &self,
        model_path: &Path,
    ) -> Result<usize, MetalError> {
        let cores = self.gpu_core_count;
        let model_size = ModelSize::from_path(model_path)?;
        Ok(match (model_size, cores) {
            (ModelSize::Large, c) if c > 20 => 32,
            (ModelSize::Large, c) if c > 10 => 16,
            (ModelSize::Large, _) => 8,
            (ModelSize::Small, c) if c > 20 => 256,
            (ModelSize::Small, c) if c > 10 => 128,
            (ModelSize::Small, _) => 64,
        })
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
        let sparse_page_size = self.sparse_heap_pool.borrow().page_size();
        let context = self.weak_self.upgrade().ok_or(MetalError::CannotCreateBuffer)?;
        MetalSparseBuffer::new(context, capacity, sparse_page_size)
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

    fn sparse_buffers_supported(&self) -> bool {
        self.device.supports_placement_sparse_resources()
    }
}

/// GPU size class used to key shape heuristics. Fleet sweep (June 2026,
/// A18 Pro 5c / M1 8c / M2 Pro 19c / M4 Pro 20c / M3 Max 40c / M4 Max 40c /
/// M5 Max) showed the deep-k wide-row tile only pays off on Max/Ultra-class
/// parts; smaller GPUs always prefer single-row tiles for batch-1 decode.
/// G13 (M1/A14 generation) additionally prefers unsplit wide-row tiles on
/// huge-n matrices (lm_head: +6.5% for SG8_KS1_R4 over KS8_R1, confirmed
/// same-run A/B; M2 Pro / M4 Pro / A18 Pro show no such preference).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDeviceTier {
    /// Phone-class and base/Pro Mac GPUs (< 30 cores), Apple9+ families
    /// (M3/M4/A17 and newer): prefer small SG2/SG4 quant tiles like the
    /// Wide tier.
    Compact,
    /// M2/A15/A16 generation compacts (Apple8, pre-Apple9): prefer SG8
    /// quant tiles with reduced R.
    CompactG14,
    /// M1/A14 generation (G13): compact and pre-Apple8.
    CompactG13,
    /// Max/Ultra-class GPUs (>= 30 cores).
    Wide,
}

impl MetalContext {
    /// GPU size/generation class used to key kernel dispatch heuristics
    /// (see `gemv::kernel`); detected once from core count + Metal family.
    pub fn gpu_device_tier(&self) -> GpuDeviceTier {
        // Test/debug override, e.g. UZU_GPU_TIER=compact_g13.
        if let Ok(name) = std::env::var("UZU_GPU_TIER") {
            match name.as_str() {
                "wide" => return GpuDeviceTier::Wide,
                "compact" => return GpuDeviceTier::Compact,
                "compact_g14" => return GpuDeviceTier::CompactG14,
                "compact_g13" => return GpuDeviceTier::CompactG13,
                _ => {},
            }
        }
        gpu_device_tier_for(self.gpu_core_count(), self.supports_apple8_family(), self.supports_apple9_family())
    }
}

fn gpu_device_tier_for(
    gpu_core_count: u32,
    supports_apple8_family: bool,
    supports_apple9_family: bool,
) -> GpuDeviceTier {
    if gpu_core_count >= 30 {
        GpuDeviceTier::Wide
    } else if !supports_apple8_family {
        GpuDeviceTier::CompactG13
    } else if !supports_apple9_family {
        GpuDeviceTier::CompactG14
    } else {
        GpuDeviceTier::Compact
    }
}

#[cfg(test)]
mod gpu_tier_tests {
    use super::*;

    #[test]
    fn gpu_device_tier_detection() {
        assert_eq!(gpu_device_tier_for(5, true, true), GpuDeviceTier::Compact); // A18 Pro
        assert_eq!(gpu_device_tier_for(8, false, false), GpuDeviceTier::CompactG13); // M1
        assert_eq!(gpu_device_tier_for(10, true, false), GpuDeviceTier::CompactG14); // M2
        assert_eq!(gpu_device_tier_for(19, true, false), GpuDeviceTier::CompactG14); // M2 Pro
        assert_eq!(gpu_device_tier_for(20, true, true), GpuDeviceTier::Compact); // M4 Pro
        assert_eq!(gpu_device_tier_for(40, true, true), GpuDeviceTier::Wide); // M3/M4 Max
        assert_eq!(gpu_device_tier_for(32, false, false), GpuDeviceTier::Wide); // M1 Max stays wide
    }
}
