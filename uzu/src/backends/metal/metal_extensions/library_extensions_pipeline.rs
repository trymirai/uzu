use metal::{
    ComputePipelineState, Device, FunctionConstantValues, Library,
    foreign_types::{ForeignType, ForeignTypeRef},
};

use crate::backends::metal::error::{LibraryError, MTLError};

/// Extensions for Library to create compute pipeline states
pub trait LibraryPipelineExtensions {
    /// Creates a compute pipeline state for a named function in the library.
    /// Optionally accepts function constants.
    fn compute_pipeline_state(
        &self,
        function_name: &str,
        constants: Option<&FunctionConstantValues>,
    ) -> Result<ComputePipelineState, MTLError>;

    /// Creates a compute pipeline state for a named function and returns argument reflection information.
    /// Optionally accepts function constants.
    fn compute_pipeline_state_with_reflection(
        &self,
        function_name: &str,
        constants: Option<&FunctionConstantValues>,
    ) -> Result<(ComputePipelineState, Vec<String>), MTLError>;
}

impl LibraryPipelineExtensions for Library {
    fn compute_pipeline_state(
        &self,
        function_name: &str,
        constants: Option<&FunctionConstantValues>,
    ) -> Result<ComputePipelineState, MTLError> {
        // Get the function from the library
        let function = match constants {
            Some(const_values) => {
                // Clone constants since get_function takes ownership
                let constants_owned = const_values.clone();
                self.get_function(function_name, Some(constants_owned))
            },
            None => self.get_function(function_name, None),
        }
        .map_err(|_| MTLError::Library(LibraryError::FunctionCreationFailed))?;

        // Safely retain the device to balance the retain/release count
        let device = self.device().to_owned();

        // Create the pipeline state
        device.new_compute_pipeline_state_with_function(&function).map_err(
            |e| {
                MTLError::Library(LibraryError::Custom(format!(
                    "Pipeline state creation failed: {}",
                    e
                )))
            },
        )
    }

    fn compute_pipeline_state_with_reflection(
        &self,
        function_name: &str,
        constants: Option<&FunctionConstantValues>,
    ) -> Result<(ComputePipelineState, Vec<String>), MTLError> {
        // Get the function from the library
        let function = match constants {
            Some(const_values) => {
                // Clone constants since get_function takes ownership
                let constants_owned = const_values.clone();
                self.get_function(function_name, Some(constants_owned))
            },
            None => self.get_function(function_name, None),
        }
        .map_err(|_| MTLError::Library(LibraryError::FunctionCreationFailed))?;

        // Safely retain the device to balance the retain/release count so that
        // the underlying Objective-C object is not over-released when the
        // temporary `Device` created here is dropped.
        let device = self.device().to_owned();

        // Create a descriptor for the pipeline
        let descriptor = metal::ComputePipelineDescriptor::new();
        descriptor.set_compute_function(Some(&function));

        // Configure reflection options to get argument info
        let reflection_options = metal::MTLPipelineOption::ArgumentInfo
            | metal::MTLPipelineOption::BufferTypeInfo;

        // Create the pipeline state with reflection
        let (pipeline_state, reflection) = device
            .new_compute_pipeline_state_with_reflection(
                &descriptor,
                reflection_options,
            )
            .map_err(|e| {
                MTLError::Library(LibraryError::Custom(format!(
                    "Pipeline state creation with reflection failed: {}",
                    e
                )))
            })?;

        // Extract argument names from reflection
        let args = reflection.arguments();
        let mut arg_names = Vec::new();

        for i in 0..args.count() {
            if let Some(arg) = args.object_at(i) {
                arg_names.push(arg.name().to_string());
            }
        }

        Ok((pipeline_state, arg_names))
    }
}
