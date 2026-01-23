use metal::MTLCommandEncoderExt;

use crate::backends::metal::{
    KernelDataType, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLContext, MTLError,
    ProtocolObject, Retained,
    metal_extensions::{
        ComputeEncoderConditional, ComputeEncoderDispatch,
        ComputeEncoderSetValue,
    },
};

#[derive(Debug)]
pub struct TensorAddBias {
    pipeline_state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl TensorAddBias {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, MTLError> {
        let function_name =
            format!("tensorAddBias_{}", data_type.function_name_suffix());
        let pipeline_state =
            context.compute_pipeline_state(&function_name, None)?;
        Ok(Self {
            pipeline_state,
        })
    }

    pub fn encode_into_command_buffer(
        &self,
        input: &ProtocolObject<dyn MTLBuffer>,
        bias: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        num_cols: usize,
        total_len: usize,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        predicate: Option<&ProtocolObject<dyn MTLBuffer>>,
    ) {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        self.encode_with_encoder(
            input, bias, output, num_cols, total_len, &encoder, predicate,
        );
        encoder.end_encoding();
    }

    pub fn encode_with_encoder(
        &self,
        input: &ProtocolObject<dyn MTLBuffer>,
        bias: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        num_cols: usize,
        total_len: usize,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        predicate: Option<&ProtocolObject<dyn MTLBuffer>>,
    ) {
        encoder.condition(
            predicate,
            || {
                encoder.set_label(Some("Tensor Add Bias"));
                encoder.set_compute_pipeline_state(&self.pipeline_state);
                encoder.set_buffer(Some(input), 0, 0);
                encoder.set_buffer(Some(bias), 0, 1);
                encoder.set_buffer(Some(output), 0, 2);
                let num_cols_i32 = num_cols as i32;
                let total_len_i32 = total_len as i32;
                encoder.set_value(&num_cols_i32, 3);
                encoder.set_value(&total_len_i32, 4);
                encoder.dispatch_1d_exactly(
                    &self.pipeline_state,
                    total_len,
                    None,
                );
            },
            None::<fn()>,
        );
    }
}
