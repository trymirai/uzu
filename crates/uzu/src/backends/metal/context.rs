#![allow(dead_code)]
include!(concat!(env!("OUT_DIR"), "/metal_lib.rs"));

use metal::{
    CommandQueue as MTLCommandQueue,
    ComputePipelineState as MTLComputePipelineState, Device as MTLDevice,
    FunctionConstantValues, Library as MTLLibrary,
};

use super::{
    MetalArray, MetalBackend, error::MTLError,
    metal_extensions::LibraryPipelineExtensions,
};
use crate::{
    DataType, array::array_size_in_bytes, backends::Context,
    env_utils::MetalEnvVar,
};

pub struct MTLContext {
    pub device: MTLDevice,
    pub command_queue: MTLCommandQueue,
    library: MTLLibrary,
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
        })
    }

    pub fn compute_pipeline_state(
        &self,
        function_name: &str,
        constants: Option<&FunctionConstantValues>,
    ) -> Result<MTLComputePipelineState, MTLError> {
        self.library.compute_pipeline_state(function_name, constants)
    }

    pub fn compute_pipeline_state_with_reflection(
        &self,
        function_name: &str,
        constants: Option<&FunctionConstantValues>,
    ) -> Result<(MTLComputePipelineState, Vec<String>), MTLError> {
        self.library
            .compute_pipeline_state_with_reflection(function_name, constants)
    }

    pub fn debug_active(&self) -> bool {
        MetalEnvVar::DeviceWrapperType.is_enabled()
    }
}

impl Context<MetalBackend> for MTLContext {
    fn default() -> Option<Self> {
        let mtl_device = metal::Device::system_default()?;
        let mtl_command_queue = mtl_device.new_command_queue();
        MTLContext::new(mtl_device, mtl_command_queue).ok()
    }

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
