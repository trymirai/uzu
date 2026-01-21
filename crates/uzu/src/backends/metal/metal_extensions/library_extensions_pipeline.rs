use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSError;

use crate::backends::metal::{
    MTLBindingExt, MTLComputePipelineDescriptor, MTLComputePipelineState,
    MTLDevice, MTLDeviceExt, MTLFunctionConstantValues, MTLLibrary,
    MTLLibraryExt, MTLPipelineOption,
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

    /// Creates a compute pipeline state for a named function and returns argument reflection information.
    /// Optionally accepts function constants.
    fn compute_pipeline_state_with_reflection(
        &self,
        function_name: &str,
        constants: Option<&MTLFunctionConstantValues>,
    ) -> Result<
        (Retained<ProtocolObject<dyn MTLComputePipelineState>>, Vec<String>),
        MTLError,
    >;
}

impl LibraryPipelineExtensions for ProtocolObject<dyn MTLLibrary> {
    fn compute_pipeline_state(
        &self,
        function_name: &str,
        constants: Option<&MTLFunctionConstantValues>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MTLError>
    {
        // Get the function from the library
        let function = match constants {
            Some(const_values) => {
                let mut error: *mut NSError = std::ptr::null_mut();
                unsafe {
                    self.new_function_with_name_constant_values_error(
                        function_name,
                        const_values,
                        &mut error,
                    )
                }
            },
            None => self.new_function_with_name(function_name),
        }
        .ok_or(MTLError::Library(LibraryError::FunctionCreationFailed))?;

        // Get the device from the library
        let device = self.device();

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
        constants: Option<&MTLFunctionConstantValues>,
    ) -> Result<
        (Retained<ProtocolObject<dyn MTLComputePipelineState>>, Vec<String>),
        MTLError,
    > {
        // Get the function from the library
        let function = match constants {
            Some(const_values) => {
                let mut error: *mut NSError = std::ptr::null_mut();
                unsafe {
                    self.new_function_with_name_constant_values_error(
                        function_name,
                        const_values,
                        &mut error,
                    )
                }
            },
            None => self.new_function_with_name(function_name),
        }
        .ok_or(MTLError::Library(LibraryError::FunctionCreationFailed))?;

        // Get the device from the library
        let device = self.device();

        // Create a descriptor for the pipeline
        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.set_compute_function(Some(&function));

        // Configure reflection options to get argument info
        let reflection_options = MTLPipelineOption::BINDING_INFO
            | MTLPipelineOption::BUFFER_TYPE_INFO;

        // Create the pipeline state with reflection
        let (pipeline_state, reflection) = device
            .new_compute_pipeline_state_with_descriptor(
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
        let mut arg_names = Vec::new();
        if let Some(ref reflection) = reflection {
            use objc2::{msg_send, rc::Retained, runtime::NSObject};
            use objc2_foundation::{NSArray, NSString};
            let arguments: Option<Retained<NSArray<NSObject>>> =
                unsafe { msg_send![reflection, arguments] };
            if let Some(arguments) = arguments {
                for argument in arguments.iter() {
                    let name: Retained<NSString> =
                        unsafe { msg_send![&*argument, name] };
                    arg_names.push(name.to_string());
                }
            }
        }

        Ok((pipeline_state, arg_names))
    }
}

impl LibraryPipelineExtensions for Retained<ProtocolObject<dyn MTLLibrary>> {
    fn compute_pipeline_state(
        &self,
        function_name: &str,
        constants: Option<&MTLFunctionConstantValues>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MTLError>
    {
        (**self).compute_pipeline_state(function_name, constants)
    }

    fn compute_pipeline_state_with_reflection(
        &self,
        function_name: &str,
        constants: Option<&MTLFunctionConstantValues>,
    ) -> Result<
        (Retained<ProtocolObject<dyn MTLComputePipelineState>>, Vec<String>),
        MTLError,
    > {
        (**self)
            .compute_pipeline_state_with_reflection(function_name, constants)
    }
}
