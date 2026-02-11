use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::backends::metal::{
    MTLComputePipelineState, MTLDeviceExt, MTLFunctionConstantValues, MTLLibrary, MTLLibraryExt,
    error::{LibraryError, MTLError},
};

/// Extensions for Library to create compute pipeline states
pub trait LibraryPipelineExtensions {
    /// Creates a compute pipeline state for a named function in the library.
    /// Optionally accepts function constants.
    fn compute_pipeline_state(
        &self,
        function_name: &str,
        constants: Option<&MTLFunctionConstantValues>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MTLError>;
}

impl LibraryPipelineExtensions for ProtocolObject<dyn MTLLibrary> {
    fn compute_pipeline_state(
        &self,
        function_name: &str,
        constants: Option<&MTLFunctionConstantValues>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MTLError> {
        let function = match constants {
            Some(const_values) => {
                self.new_function_with_name_constant_values_error(function_name, const_values, std::ptr::null_mut())
            },
            None => self.new_function_with_name(function_name),
        }
        .ok_or(MTLError::Library(LibraryError::FunctionCreationFailed))?;

        let device = self.device();

        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| MTLError::Library(LibraryError::Custom(format!("Pipeline state creation failed: {}", e))))
    }
}
