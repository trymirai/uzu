use metal::{
    MTLComputePipelineDescriptor, MTLComputePipelineState, MTLDeviceExt, MTLFunctionConstantValues, MTLLibrary,
    MTLLibraryExt, MTLPipelineOption,
};
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::backends::metal::error::MetalError;

/// Extensions for Library to create compute pipeline states
pub trait LibraryPipelineExtensions {
    /// Creates a compute pipeline state for a named function in the library.
    /// Optionally accepts function constants.
    fn compute_pipeline_state(
        &self,
        function_name: &str,
        constants: Option<&MTLFunctionConstantValues>,
        label: Option<&str>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError>;
}

impl LibraryPipelineExtensions for ProtocolObject<dyn MTLLibrary> {
    fn compute_pipeline_state(
        &self,
        function_name: &str,
        constants: Option<&MTLFunctionConstantValues>,
        label: Option<&str>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        let function = match constants {
            Some(const_values) => self
                .new_function_with_name_constant_values(function_name, const_values)
                .map_err(|error| MetalError::CannotCreateFunction(format!("{function_name}: {error}")))?,
            None => self
                .new_function_with_name(function_name)
                .ok_or_else(|| MetalError::CannotCreateFunction(function_name.to_owned()))?,
        };

        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.set_compute_function(Some(&function));
        descriptor.set_label(label);

        self.device()
            .new_compute_pipeline_state_with_descriptor(&descriptor, MTLPipelineOption::NONE)
            .map(|(pipeline, _)| pipeline)
            .map_err(|nserror| MetalError::CannotCreatePipelineState {
                function_name: function_name.to_owned(),
                error: nserror.to_string(),
            })
    }
}
