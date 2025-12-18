use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use crate::{
    DataType,
    backends::metal::{MTLContext, MTLError},
    config::Activation,
};

pub struct ActivationKernel {
    pipeline: MTLComputePipelineState,
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
        let (pipeline, _) =
            context.compute_pipeline_state_with_reflection(fn_name, None)?;
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
        encoder: &ComputeCommandEncoderRef,
        activation: &Activation,
        input_buffer: &MTLBuffer,
        output_buffer: &MTLBuffer,
        n: usize,
    ) -> Result<(), MTLError> {
        let act_code = Self::act_code(activation);
        let threads_per_tg = 256u64;
        let threadgroups = (n as u64 + threads_per_tg - 1) / threads_per_tg;

        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(output_buffer), 0);
        encoder.set_bytes(
            2,
            size_of::<i32>() as u64,
            &(n as i32) as *const i32 as *const _,
        );
        encoder.set_bytes(
            3,
            size_of::<u16>() as u64,
            &act_code as *const u16 as *const _,
        );

        encoder.dispatch_thread_groups(
            MTLSize::new(threadgroups, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
        Ok(())
    }
}

