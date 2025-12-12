use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef, ComputePipelineState,
    MTLSize,
};

use crate::{
    DataType,
    backends::metal::{MTLContext, MTLError},
};

pub struct PoolingKernel {
    cls_pipeline: ComputePipelineState,
    mean_pipeline: ComputePipelineState,
}

impl PoolingKernel {
    fn kernel_name_for_type(
        data_type: DataType,
        pooling_type: &str,
    ) -> Result<String, MTLError> {
        let suffix = match data_type {
            DataType::F16 => "f16",
            DataType::F32 => "f32",
            DataType::BF16 => "bf16",
            other => {
                return Err(MTLError::Generic(format!(
                    "Unsupported dtype for pooling: {:?}",
                    other
                )));
            },
        };
        Ok(format!("pool_{}_{}", pooling_type, suffix))
    }

    pub fn new(
        context: &MTLContext,
        data_type: DataType,
    ) -> Result<Self, MTLError> {
        let cls_fn_name = Self::kernel_name_for_type(data_type, "cls")?;
        let mean_fn_name = Self::kernel_name_for_type(data_type, "mean")?;

        let cls_pipeline =
            context.compute_pipeline_state(&cls_fn_name, None)?;
        let mean_pipeline =
            context.compute_pipeline_state(&mean_fn_name, None)?;

        Ok(Self {
            cls_pipeline,
            mean_pipeline,
        })
    }

    pub fn encode_cls(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &MTLBuffer,
        output_buffer: &MTLBuffer,
        batch_size: i32,
        seq_len: i32,
        hidden_dim: i32,
    ) -> Result<(), MTLError> {
        encoder.set_compute_pipeline_state(&self.cls_pipeline);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(output_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<i32>() as u64,
            &seq_len as *const i32 as *const _,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<i32>() as u64,
            &hidden_dim as *const i32 as *const _,
        );

        let threads_per_tg = MTLSize::new(16, 16, 1);
        let grid = MTLSize::new(hidden_dim as u64, batch_size as u64, 1);
        encoder.dispatch_threads(grid, threads_per_tg);
        Ok(())
    }

    pub fn encode_mean(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &MTLBuffer,
        output_buffer: &MTLBuffer,
        batch_size: i32,
        seq_len: i32,
        hidden_dim: i32,
    ) -> Result<(), MTLError> {
        encoder.set_compute_pipeline_state(&self.mean_pipeline);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(output_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<i32>() as u64,
            &seq_len as *const i32 as *const _,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<i32>() as u64,
            &hidden_dim as *const i32 as *const _,
        );

        let threads_per_tg = MTLSize::new(16, 16, 1);
        let grid = MTLSize::new(hidden_dim as u64, batch_size as u64, 1);
        encoder.dispatch_threads(grid, threads_per_tg);
        Ok(())
    }
}
