use metal::{
    MTL4Compiler, MTL4CompilerExt, MTL4ComputePipelineDescriptor, MTL4LibraryFunctionDescriptor,
    MTL4SpecializedFunctionDescriptor, MTLComputePipelineState, MTLFunctionConstantValues, MTLLibrary,
    MTLNewComputePipelineStateCompletionHandler,
};
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::backends::metal::error::MetalError;

type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Extensions for a Metal 4 compiler to schedule compute pipeline creation.
pub trait MTL4CompilerExtensions {
    fn schedule_compute_pipeline_state(
        &self,
        library: &ProtocolObject<dyn MTLLibrary>,
        function_name: &str,
        constants: Option<&MTLFunctionConstantValues>,
        completion: impl Fn(Result<ComputePipelineState, MetalError>) + Send + Sync + 'static,
    );
}

impl MTL4CompilerExtensions for ProtocolObject<dyn MTL4Compiler> {
    fn schedule_compute_pipeline_state(
        &self,
        library: &ProtocolObject<dyn MTLLibrary>,
        function_name: &str,
        constants: Option<&MTLFunctionConstantValues>,
        completion: impl Fn(Result<ComputePipelineState, MetalError>) + Send + Sync + 'static,
    ) {
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

        let function_name = function_name.to_owned();
        let completion_handler = MTLNewComputePipelineStateCompletionHandler::new(move |pipeline, error| {
            completion(pipeline.ok_or_else(|| MetalError::CannotCreatePipelineState {
                function_name: function_name.clone(),
                error: error.map_or_else(
                    || "Metal returned neither a pipeline nor an error".to_owned(),
                    |error| error.to_string(),
                ),
            }));
        });
        self.new_compute_pipeline_state_with_descriptor_compiler_task_options_completion_handler(
            &pipeline_descriptor,
            None,
            completion_handler,
        );
    }
}
