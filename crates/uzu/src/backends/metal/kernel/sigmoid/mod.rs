use crate::{
    DataType,
    backends::metal::{
        ComputeEncoderSetValue, MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState,
        MTLContext, MTLError, MTLSize, ProtocolObject, Retained,
    },
};

pub struct SigmoidKernel {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl SigmoidKernel {
    fn kernel_name_for_type(
        data_type: DataType
    ) -> Result<&'static str, MTLError> {
        match data_type {
            DataType::F16 => Ok("apply_sigmoid_f16"),
            DataType::F32 => Ok("apply_sigmoid_f32"),
            DataType::BF16 => Ok("apply_sigmoid_bf16"),
            other => Err(MTLError::Generic(format!(
                "Unsupported dtype for sigmoid: {:?}",
                other
            ))),
        }
    }

    pub fn new(
        context: &MTLContext,
        data_type: DataType,
    ) -> Result<Self, MTLError> {
        let fn_name = Self::kernel_name_for_type(data_type)?;
        let pipeline = context.compute_pipeline_state(fn_name, None)?;
        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input_buffer: &ProtocolObject<dyn MTLBuffer>,
        output_buffer: &ProtocolObject<dyn MTLBuffer>,
        total_elements: i32,
    ) -> Result<(), MTLError> {
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(Some(input_buffer), 0, 0);
        encoder.set_buffer(Some(output_buffer), 0, 1);
        encoder.set_value(&total_elements, 2);

        let threads_per_tg = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(total_elements as usize, 1, 1);
        encoder.dispatch_threads(grid, threads_per_tg);
        Ok(())
    }
}
