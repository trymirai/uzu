#![allow(dead_code)]
include!(concat!(env!("OUT_DIR"), "/metal_lib.rs"));

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use metal::{
    CommandQueue as MTLCommandQueue,
    ComputePipelineState as MTLComputePipelineState, Device as MTLDevice,
    FunctionConstantValues, Library as MTLLibrary,
};

use super::{
    MetalArray, error::MTLError, metal_extensions::LibraryPipelineExtensions,
};
use crate::{DataType, DeviceContext, array::array_size_in_bytes};

pub struct MTLContext {
    pub device: MTLDevice,
    pub command_queue: MTLCommandQueue,
    library: MTLLibrary,
    pipeline_cache: RefCell<HashMap<String, MTLComputePipelineState>>,
}

impl MTLContext {
    pub fn new(
        device: MTLDevice,
        command_queue: MTLCommandQueue,
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

        Ok(Self {
            device,
            command_queue,
            library,
            pipeline_cache: RefCell::new(HashMap::new()),
        })
    }

    pub fn compute_pipeline_state(
        &self,
        function_name: &str,
        constants: Option<&FunctionConstantValues>,
    ) -> Result<MTLComputePipelineState, MTLError> {
        // Only cache pipelines without constants
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
        // Only cache pipelines without constants
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

            let buffer = self.device.new_buffer(
                buffer_size_bytes as u64,
                metal::MTLResourceOptions::StorageModeShared,
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
