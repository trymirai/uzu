use crate::{
    DataType,
    backends::metal::{
        ComputeEncoderSetValue, MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState,
        MTLContext, MTLError, MTLSize, ProtocolObject, Retained,
    },
    config::Activation,
};

pub struct ActivationKernel {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl ActivationKernel {
    fn kernel_name_for_type(
        data_type: DataType
    ) -> Result<&'static str, MTLError> {
        match data_type {
            DataType::F16 => Ok("activation_f16"),
            DataType::F32 => Ok("activation_f32"),
            DataType::BF16 => Ok("activation_bf16"),
            other => Err(MTLError::Generic(format!(
                "Unsupported dtype for activation: {:?}",
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

    fn act_code(act: &Activation) -> u16 {
        match act {
            Activation::SiLU {
                ..
            } => 0,
            Activation::Gelu => 1,
            Activation::Identity => {
                panic!("Identity activation is not supported for kernel")
            },
        }
    }

    pub fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        activation: &Activation,
        input_buffer: &ProtocolObject<dyn MTLBuffer>,
        output_buffer: &ProtocolObject<dyn MTLBuffer>,
        n: usize,
    ) -> Result<(), MTLError> {
        let act_code = Self::act_code(activation);
        let threads_per_tg = 256u64;
        let threadgroups = (n as u64 + threads_per_tg - 1) / threads_per_tg;

        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(Some(input_buffer), 0, 0);
        encoder.set_buffer(Some(output_buffer), 0, 1);
        encoder.set_value(&(n as i32), 2);
        encoder.set_value(&act_code, 3);

        encoder.dispatch_threadgroups(
            MTLSize::new(threadgroups as usize, 1, 1),
            MTLSize::new(threads_per_tg as usize, 1, 1),
        );
        Ok(())
    }
}
