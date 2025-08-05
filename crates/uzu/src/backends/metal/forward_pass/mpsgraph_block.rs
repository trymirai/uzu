use mpsgraph::{
    CommandBuffer as MPSCommandBuffer, Executable,
    ExecutableExecutionDescriptor, TensorData,
};
use objc2::rc::Retained;

use super::{
    encodable_with_state::EncodableWithState, io_arrays::IOArrays,
    state::ForwardPassState,
};
use crate::backends::metal::forward_pass::encodable_with_state::EncodingParameters;

pub struct MPSGraphBlock {
    executable: Retained<Executable>,
    execution_descriptor: Retained<ExecutableExecutionDescriptor>,
    arguments: IOArrays,
}

impl MPSGraphBlock {
    pub fn new(
        executable: Retained<Executable>,
        execution_descriptor: Retained<ExecutableExecutionDescriptor>,
        arguments: IOArrays,
    ) -> Self {
        Self {
            executable,
            execution_descriptor,
            arguments,
        }
    }

    // fn load(path: &Path) -> Result<Self, ExecutableBlockError> {
    //     let signature =
    //         from_reader(fs::File::open(path.join("signature.json"))?)?;
    //     let executable =
    //         Executable::from_serialized_package(path, None).unwrap();
    //     let execution_descriptor = ExecutableExecutionDescriptor::new();

    //     Ok(Self {
    //         signature,
    //         executable,
    //         execution_descriptor,
    //     })
    // }

    // fn save(
    //     &self,
    //     path: &Path,
    // ) -> Result<(), ExecutableBlockError> {
    //     fs::create_dir(path)?;

    //     self.executable.serialize_to_url(path, &SerializationDescriptor::new());

    //     let writer = fs::File::create(path.join("signature.json"))?;
    //     to_writer(writer, &self.signature)?;

    //     Ok(())
    // }
}

impl EncodableWithState for MPSGraphBlock {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        unsafe {
            let feeds = self.arguments.get_mpsgraph_feeds(state);

            let inputs_refs: Vec<&TensorData> =
                feeds.inputs.iter().map(|retained_td| &**retained_td).collect();
            let inputs_slice: &[&TensorData] = &inputs_refs;

            let outputs_refs: Vec<&TensorData> = feeds
                .outputs
                .iter()
                .map(|retained_td| &**retained_td)
                .collect();
            let outputs_slice: &[&TensorData] = &outputs_refs;

            let maybe_outputs_slice = if outputs_slice.is_empty() {
                None
            } else {
                Some(outputs_slice)
            };

            let execution_descriptor = self.execution_descriptor.clone();
            execution_descriptor
                .set_enable_commit_and_continue(parameters.enable_commit);

            let root_command_buffer =
                command_buffer.root_command_buffer().to_owned();
            let _ = self.executable.encode_to_command_buffer(
                command_buffer,
                inputs_slice,
                maybe_outputs_slice,
                Some(&execution_descriptor),
            );
            if parameters.wait_until_completed {
                command_buffer.commit_and_continue();
                root_command_buffer.wait_until_completed();
            }
        }
    }
}
