use crate::{
    DataType,
    backends::metal::{
        ComputeEncoderSetValue, MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState,
        MTLContext, MTLError, MTLSize, ProtocolObject, Retained,
    },
};

pub struct PoolingKernel {
    cls_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    mean_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
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
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input_buffer: &ProtocolObject<dyn MTLBuffer>,
        output_buffer: &ProtocolObject<dyn MTLBuffer>,
        batch_size: i32,
        seq_len: i32,
        hidden_dim: i32,
    ) -> Result<(), MTLError> {
        encoder.set_compute_pipeline_state(&self.cls_pipeline);
        encoder.set_buffer(Some(input_buffer), 0, 0);
        encoder.set_buffer(Some(output_buffer), 0, 1);
        encoder.set_value(&seq_len, 2);
        encoder.set_value(&hidden_dim, 3);

        let threads_per_tg = MTLSize::new(16, 16, 1);
        let grid = MTLSize::new(hidden_dim as usize, batch_size as usize, 1);
        encoder.dispatch_threads(grid, threads_per_tg);
        Ok(())
    }

    pub fn encode_mean(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input_buffer: &ProtocolObject<dyn MTLBuffer>,
        output_buffer: &ProtocolObject<dyn MTLBuffer>,
        batch_size: i32,
        seq_len: i32,
        hidden_dim: i32,
    ) -> Result<(), MTLError> {
        encoder.set_compute_pipeline_state(&self.mean_pipeline);
        encoder.set_buffer(Some(input_buffer), 0, 0);
        encoder.set_buffer(Some(output_buffer), 0, 1);
        encoder.set_value(&seq_len, 2);
        encoder.set_value(&hidden_dim, 3);

        let threads_per_tg = MTLSize::new(16, 16, 1);
        let grid = MTLSize::new(hidden_dim as usize, batch_size as usize, 1);
        encoder.dispatch_threads(grid, threads_per_tg);
        Ok(())
    }
}
