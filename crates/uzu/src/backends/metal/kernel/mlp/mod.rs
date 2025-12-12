use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef, ComputePipelineState,
    MTLSize,
};

use crate::{
    DataType,
    backends::metal::{MTLContext, MTLError},
    config::Activation,
};

pub struct MlpGateActMulKernel {
    pipeline: ComputePipelineState,
}

pub struct MlpGateActMulEncodable {
    kernel: MlpGateActMulKernel,
    activation: Activation,
    hidden_dim: usize,
}

impl MlpGateActMulEncodable {
    pub fn new(
        context: &MTLContext,
        data_type: DataType,
        activation: Activation,
        hidden_dim: usize,
    ) -> Result<Self, MTLError> {
        let kernel = MlpGateActMulKernel::new(context, data_type)?;
        Ok(Self {
            kernel,
            activation,
            hidden_dim,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        fused_up: &MTLBuffer,
        hidden: &MTLBuffer,
        m: i32,
    ) -> Result<(), MTLError> {
        self.kernel.encode(
            encoder,
            &self.activation,
            fused_up,
            hidden,
            m,
            self.hidden_dim as i32,
        )
    }
}

impl MlpGateActMulKernel {
    fn kernel_name_for_type(data_type: DataType) -> Result<&'static str, MTLError> {
        match data_type {
            DataType::F16 => Ok("mlp_activation_mul_f16"),
            DataType::F32 => Ok("mlp_activation_mul_f32"),
            DataType::BF16 => Ok("mlp_activation_mul_bf16"),
            other => Err(MTLError::Generic(format!(
                "Unsupported dtype for MLP activation: {:?}",
                other
            ))),
        }
    }

    pub fn new(context: &MTLContext, data_type: DataType) -> Result<Self, MTLError> {
        let fn_name = Self::kernel_name_for_type(data_type)?;
        let pipeline = context.compute_pipeline_state(fn_name, None)?;
        Ok(Self { pipeline })
    }

    fn act_code(act: &Activation) -> u16 {
        match act {
            Activation::SiLU { .. } => 0,
            Activation::Gelu => 1,
            Activation::Identity => {
                panic!("Identity activation is not supported for MLP kernels")
            },
        }
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        activation: &Activation,
        fused_up_buffer: &MTLBuffer,
        hidden_buffer: &MTLBuffer,
        m: i32,
        h: i32,
    ) -> Result<(), MTLError> {
        let act_code = Self::act_code(activation);
        let threads_per_tg = MTLSize::new(64, 1, 1);
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(fused_up_buffer), 0);
        encoder.set_buffer(1, Some(hidden_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<i32>() as u64,
            &h as *const i32 as *const _,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<i32>() as u64,
            &m as *const i32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u16>() as u64,
            &act_code as *const u16 as *const _,
        );

        let grid = MTLSize::new(h as u64, m as u64, 1);
        encoder.dispatch_threads(grid, threads_per_tg);
        Ok(())
    }
}
