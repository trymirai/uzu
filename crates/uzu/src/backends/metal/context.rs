#![allow(dead_code)]
include!(concat!(env!("OUT_DIR"), "/metal_lib.rs"));

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use metal::{
    Buffer as MTLBuffer, CommandQueue as MTLCommandQueue,
    ComputePipelineState as MTLComputePipelineState, Device as MTLDevice,
    FunctionConstantValues, Library as MTLLibrary, MTLResourceOptions,
    MTLStorageMode,
};

use super::{
    MetalArray,
    buffer_allocator::{BufferAllocator, FallbackHeapAllocator},
    error::MTLError,
    fence::FenceRegistry,
    metal_extensions::LibraryPipelineExtensions,
};
use crate::{DataType, DeviceContext, array::array_size_in_bytes};

pub struct MTLContext {
    pub device: MTLDevice,
    pub command_queue: MTLCommandQueue,
    library: MTLLibrary,
    pipeline_cache: RefCell<HashMap<String, MTLComputePipelineState>>,
    heap_allocator: Option<FallbackHeapAllocator>,
    fence_registry: Rc<FenceRegistry>,
}

impl MTLContext {
    pub fn new(
        device: MTLDevice,
        command_queue: MTLCommandQueue,
    ) -> Result<Self, MTLError> {
        Self::new_with_heap(device, command_queue, None)
    }

    pub fn new_with_heap(
        device: MTLDevice,
        command_queue: MTLCommandQueue,
        heap_size_bytes: Option<u64>,
    ) -> Result<Self, MTLError> {
        let library = match device.new_library_with_data(METAL_LIBRARY_DATA) {
            Ok(lib) => lib,
            Err(e) => {
                return Err(MTLError::Generic(format!(
                    "Failed to create Metal library: {}",
                    e
                )));
            },
        };

        let heap_allocator = heap_size_bytes.map(|size| {
            FallbackHeapAllocator::new(
                device.clone(),
                size,
                MTLStorageMode::Shared,
            )
        });

        let fence_registry = Rc::new(FenceRegistry::new(device.clone()));

        Ok(Self {
            device,
            command_queue,
            library,
            pipeline_cache: RefCell::new(HashMap::new()),
            heap_allocator,
            fence_registry,
        })
    }

    /// Get the fence registry for manual synchronization with untracked buffers
    pub fn fence_registry(&self) -> Rc<FenceRegistry> {
        self.fence_registry.clone()
    }

    pub fn allocate_buffer(
        &self,
        size_bytes: usize,
        options: MTLResourceOptions,
    ) -> MTLBuffer {
        if let Some(ref allocator) = self.heap_allocator {
            allocator.allocate_buffer(size_bytes, options)
        } else {
            // Even without heap, use untracked mode for consistency
            let untracked_options =
                options | MTLResourceOptions::HazardTrackingModeUntracked;
            self.device.new_buffer(size_bytes as u64, untracked_options)
        }
    }

    pub fn print_heap_stats(&self) {
        if let Some(ref allocator) = self.heap_allocator {
            let (heap_count, device_count, used, total) = allocator.stats();
            eprintln!(
                "[MTLContext] Heap stats: {} heap allocs, {} device fallbacks, \
                 used={:.1}MB / total={:.1}MB ({:.1}%)",
                heap_count,
                device_count,
                used as f64 / (1024.0 * 1024.0),
                total as f64 / (1024.0 * 1024.0),
                (used as f64 / total as f64) * 100.0
            );
        }
    }

    pub fn compute_pipeline_state(
        &self,
        function_name: &str,
        constants: Option<&FunctionConstantValues>,
    ) -> Result<MTLComputePipelineState, MTLError> {
        if constants.is_some() {
            return self
                .library
                .compute_pipeline_state(function_name, constants);
        }
        self.compute_pipeline_state_cached(function_name, function_name, None)
    }

    pub fn compute_pipeline_state_cached(
        &self,
        cache_key: &str,
        function_name: &str,
        constants: Option<&FunctionConstantValues>,
    ) -> Result<MTLComputePipelineState, MTLError> {
        if let Some(pipeline) = self.pipeline_cache.borrow().get(cache_key) {
            return Ok(pipeline.clone());
        }

        let pipeline =
            self.library.compute_pipeline_state(function_name, constants)?;

        self.pipeline_cache
            .borrow_mut()
            .insert(cache_key.to_string(), pipeline.clone());

        Ok(pipeline)
    }

    pub fn compute_pipeline_state_with_reflection(
        &self,
        function_name: &str,
        constants: Option<&FunctionConstantValues>,
    ) -> Result<(MTLComputePipelineState, Vec<String>), MTLError> {
        if constants.is_some() {
            return self.library.compute_pipeline_state_with_reflection(
                function_name,
                constants,
            );
        }
        self.compute_pipeline_state_with_reflection_cached(
            function_name,
            function_name,
            None,
        )
    }

    pub fn compute_pipeline_state_with_reflection_cached(
        &self,
        cache_key: &str,
        function_name: &str,
        constants: Option<&FunctionConstantValues>,
    ) -> Result<(MTLComputePipelineState, Vec<String>), MTLError> {
        if let Some(pipeline) = self.pipeline_cache.borrow().get(cache_key) {
            return Ok((pipeline.clone(), Vec::new()));
        }

        let (pipeline, arg_names) = self
            .library
            .compute_pipeline_state_with_reflection(function_name, constants)?;

        self.pipeline_cache
            .borrow_mut()
            .insert(cache_key.to_string(), pipeline.clone());

        Ok((pipeline, arg_names))
    }
}

impl DeviceContext for MTLContext {
    type DeviceArray = MetalArray;

    unsafe fn array_uninitialized(
        &self,
        shape: &[usize],
        data_type: DataType,
    ) -> MetalArray {
        unsafe {
            let buffer_size_bytes = array_size_in_bytes(shape, data_type);

            let buffer = self.allocate_buffer(
                buffer_size_bytes,
                MTLResourceOptions::StorageModeShared,
            );
            MetalArray::new(buffer, shape, data_type)
        }
    }
}

impl DeviceContext for Rc<MTLContext> {
    type DeviceArray = MetalArray;

    unsafe fn array_uninitialized(
        &self,
        shape: &[usize],
        data_type: DataType,
    ) -> MetalArray {
        unsafe { (**self).array_uninitialized(shape, data_type) }
    }
}
