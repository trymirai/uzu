use metal::{
    MTL4Compiler, MTL4CompilerExt, MTL4ComputePipelineDescriptor, MTL4LibraryFunctionDescriptor,
    MTL4SpecializedFunctionDescriptor, MTLComputePipelineState, MTLFunctionConstantValues, MTLLibrary,
};
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::backends::metal::error::MetalError;

/// Extensions for a Metal 4 compiler to create compute pipeline states.
pub trait CompilerPipelineExtensions {
    /// Creates a compute pipeline state for a named function in the library.
    /// Optionally specializes the function with constant values.
    fn compute_pipeline_state(
        &self,
        library: &ProtocolObject<dyn MTLLibrary>,
        function_name: &str,
        constants: Option<&MTLFunctionConstantValues>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError>;
}

impl CompilerPipelineExtensions for ProtocolObject<dyn MTL4Compiler> {
    fn compute_pipeline_state(
        &self,
        library: &ProtocolObject<dyn MTLLibrary>,
        function_name: &str,
        constants: Option<&MTLFunctionConstantValues>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        let library_function = MTL4LibraryFunctionDescriptor::new();
        library_function.set_library(Some(library));
        library_function.set_name(Some(function_name));

        let pipeline_descriptor = MTL4ComputePipelineDescriptor::new();
        match constants {
            Some(constants) => {
                let specialized_function = MTL4SpecializedFunctionDescriptor::new();
                specialized_function.set_function_descriptor(Some(&library_function));
                specialized_function.set_constant_values(Some(constants));
                pipeline_descriptor.set_compute_function_descriptor(Some(&specialized_function));
            },
            None => pipeline_descriptor.set_compute_function_descriptor(Some(&library_function)),
        }

        self.new_compute_pipeline_state_with_descriptor_compiler_task_options_error(&pipeline_descriptor, None).map_err(
            |error| MetalError::CannotCreatePipelineState {
                function_name: function_name.to_owned(),
                error: error.to_string(),
            },
        )
    }
}
