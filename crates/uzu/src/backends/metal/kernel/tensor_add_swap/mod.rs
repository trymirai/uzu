use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, CommandBuffer as MTLCommandBuffer,
    ComputePipelineState as MTLComputePipelineState, MTLCompareFunction,
    foreign_types::ForeignType,
};
use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{
    KernelDataType, MTLContext,
    metal_extensions::{ComputeEncoderConditional, ComputeEncoderDispatch},
};
use crate::{
    Array,
    backends::metal::{
        error::MTLError,
        forward_pass::{
            ArrayId, ForwardPassState,
            encodable_with_state::{EncodableWithState, EncodingParameters},
        },
    },
};

#[derive(Debug)]
pub struct TensorAddSwap {
    pipeline_state: MTLComputePipelineState,
    argument_arrays: Box<[ArrayId]>,
}

impl TensorAddSwap {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
        argument_arrays: Box<[ArrayId]>,
    ) -> Result<Self, MTLError> {
        let function_name =
            format!("tensorAddSwap_{}", data_type.function_name_suffix());

        let (pipeline_state, _argument_names) = context
            .compute_pipeline_state_with_reflection(&function_name, None)?;

        Ok(Self {
            pipeline_state,
            argument_arrays,
        })
    }

    fn encode_into_command_buffer(
        &self,
        skip_buffer: &MTLBuffer,
        main_buffer: &MTLBuffer,
        length: usize,
        command_buffer: &MTLCommandBuffer,
        predicate: Option<&MTLBuffer>,
    ) {
        let compute_encoder = command_buffer.new_compute_command_encoder();

        if let Some(p) = predicate {
            unsafe {
                compute_encoder.encode_start_if(
                    &p.as_ref(),
                    0,
                    MTLCompareFunction::NotEqual,
                    0,
                );
            }
        }

        compute_encoder.set_label("Tensor Add-Swap");

        compute_encoder.set_buffer(0, Some(skip_buffer), 0);
        compute_encoder.set_buffer(1, Some(main_buffer), 0);

        compute_encoder.set_bytes(
            2,
            size_of::<i32>() as u64,
            &(length as i32) as *const _ as *const std::ffi::c_void,
        );

        compute_encoder.dispatch_1d_exactly(&self.pipeline_state, length, None);

        if let Some(_) = predicate {
            unsafe {
                compute_encoder.encode_end_if();
            }
        }

        compute_encoder.end_encoding();
    }
}

impl EncodableWithState for TensorAddSwap {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        let arrays = state.arrays(&self.argument_arrays);
        assert_eq!(arrays.len(), 2, "TensorAddSwap expects exactly 2 arrays");

        let length = arrays[0].borrow().num_elements();

        let mut skip_array = arrays[0].borrow_mut();
        let mut main_array = arrays[1].borrow_mut();
        let skip_mtl_buffer = unsafe { skip_array.mtl_buffer() };
        let main_mtl_buffer = unsafe { main_array.mtl_buffer() };

        let retained_mtl_command_buffer =
            command_buffer.root_command_buffer().to_owned();

        self.encode_into_command_buffer(
            &skip_mtl_buffer,
            &main_mtl_buffer,
            length,
            &retained_mtl_command_buffer,
            parameters.predicate,
        );

        if parameters.wait_until_completed {
            command_buffer.commit_and_continue();
            retained_mtl_command_buffer.wait_until_completed();
        }
    }
}
