use crate::{
    DataType,
    backends::metal::{
        ComputeEncoderSetValue, MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLContext, MTLError,
        MTLSize, ProtocolObject, Retained,
    },
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RMSNormKernelType {
    Standard,
    QueryKey,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QKNormTarget {
    QueryHeads,
    KeyHeads,
    Both,
}

pub struct RMSNormKernel {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    kernel_type: RMSNormKernelType,
}

#[derive(Debug)]
pub struct RMSNormArguments<'a> {
    pub input_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub input_offset: u64,
    pub scales_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub output_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub output_offset: u64,
    pub batch_size: i32,
    pub model_dim: i32,
    pub epsilon: f32,
    pub scale_offset: f32,
}

#[derive(Debug)]
pub struct QKNormArguments<'a> {
    pub qkv_input_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub scales_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub qkv_output_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub batch_size: i32,
    pub num_q_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub epsilon: f32,
    pub scale_offset: f32,
    pub target: QKNormTarget,
}

#[derive(Debug, thiserror::Error)]
pub enum RMSNormError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error(
        "Unsupported data type combination: input={input:?}, scales={scales:?}, output={output:?}, accumulation={accumulation:?}"
    )]
    UnsupportedDataType {
        input: DataType,
        scales: DataType,
        output: DataType,
        accumulation: DataType,
    },
    #[error("Invalid kernel type for operation")]
    InvalidKernelType,
}

impl RMSNormKernel {
    pub fn new(
        context: &MTLContext,
        input_data_type: DataType,
        scales_data_type: DataType,
        output_data_type: DataType,
        accumulation_data_type: DataType,
        kernel_type: RMSNormKernelType,
    ) -> Result<Self, RMSNormError> {
        Self::new_with_mode(
            context,
            input_data_type,
            scales_data_type,
            output_data_type,
            accumulation_data_type,
            kernel_type,
            false,
        )
    }

    pub fn new_with_mode(
        context: &MTLContext,
        input_data_type: DataType,
        scales_data_type: DataType,
        output_data_type: DataType,
        accumulation_data_type: DataType,
        kernel_type: RMSNormKernelType,
        full_layer: bool,
    ) -> Result<Self, RMSNormError> {
        let function_name = Self::kernel_name_for_types(
            input_data_type,
            scales_data_type,
            output_data_type,
            accumulation_data_type,
            kernel_type,
            full_layer,
        )?;

        let pipeline = context.compute_pipeline_state(&function_name, None).map_err(RMSNormError::MetalError)?;

        Ok(Self {
            pipeline,
            kernel_type,
        })
    }

    /// Generate kernel name from data type combination
    fn kernel_name_for_types(
        input_dt: DataType,
        scales_dt: DataType,
        output_dt: DataType,
        accum_dt: DataType,
        kernel_type: RMSNormKernelType,
        full_layer: bool,
    ) -> Result<String, RMSNormError> {
        let input_suffix = Self::data_type_to_suffix(input_dt)?;
        let scales_suffix = Self::data_type_to_suffix(scales_dt)?;
        let output_suffix = Self::data_type_to_suffix(output_dt)?;
        let accum_suffix = Self::accum_type_to_suffix(accum_dt)?;

        let base_name = match kernel_type {
            RMSNormKernelType::Standard => "rms_norm",
            RMSNormKernelType::QueryKey => "qk_norm",
        };

        let mode_suffix = if full_layer {
            "_full"
        } else {
            "_norm"
        };

        Ok(format!(
            "{}_{}_{}_{}_{}{}",
            base_name, input_suffix, scales_suffix, output_suffix, accum_suffix, mode_suffix
        ))
    }

    fn data_type_to_suffix(data_type: DataType) -> Result<&'static str, RMSNormError> {
        match data_type {
            DataType::F32 => Ok("f32"),
            DataType::F16 => Ok("f16"),
            DataType::BF16 => Ok("bf16"),
            _ => Err(RMSNormError::UnsupportedDataType {
                input: data_type,
                scales: data_type,
                output: data_type,
                accumulation: data_type,
            }),
        }
    }

    fn accum_type_to_suffix(data_type: DataType) -> Result<&'static str, RMSNormError> {
        match data_type {
            DataType::F32 => Ok("f32"),
            DataType::F16 => Ok("f16"),
            _ => Err(RMSNormError::UnsupportedDataType {
                input: data_type,
                scales: data_type,
                output: data_type,
                accumulation: data_type,
            }),
        }
    }

    pub fn encode_qk_norm(
        &self,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: QKNormArguments,
    ) -> Result<(), RMSNormError> {
        if self.kernel_type != RMSNormKernelType::QueryKey {
            return Err(RMSNormError::InvalidKernelType);
        }

        compute_encoder.set_compute_pipeline_state(&self.pipeline);

        // Set buffers
        compute_encoder.set_buffer(Some(args.qkv_input_buffer), 0, 0);
        compute_encoder.set_buffer(Some(args.scales_buffer), 0, 1);
        compute_encoder.set_buffer(Some(args.qkv_output_buffer), 0, 2);

        // Set parameters
        compute_encoder.set_value(&args.batch_size, 3);
        compute_encoder.set_value(&args.num_q_heads, 4);
        compute_encoder.set_value(&args.num_kv_heads, 5);
        compute_encoder.set_value(&args.head_dim, 6);
        compute_encoder.set_value(&args.epsilon, 7);
        compute_encoder.set_value(&args.scale_offset, 8);

        // Determine which contiguous head range to normalize.
        //
        // QKV layout per token: [Q heads][K heads][V heads]
        // We only ever normalize within [Q] and/or [K].
        let (head_offset, head_count): (u32, u32) = match args.target {
            QKNormTarget::QueryHeads => (0, args.num_q_heads as u32),
            QKNormTarget::KeyHeads => (args.num_q_heads as u32, args.num_kv_heads as u32),
            QKNormTarget::Both => (0, (args.num_q_heads + args.num_kv_heads) as u32),
        };

        if args.batch_size <= 0 || args.head_dim <= 0 || head_count == 0 {
            return Ok(());
        }

        // Pass head range to the kernel.
        compute_encoder.set_value(&head_offset, 9);
        compute_encoder.set_value(&head_count, 10);

        // One SIMD-group per head, multiple heads per threadgroup.
        let simd_width = self.pipeline.thread_execution_width();
        let max_threads = self.pipeline.max_total_threads_per_threadgroup();
        let max_heads_per_threadgroup = (max_threads / simd_width).max(1);
        let heads_per_threadgroup = (head_count as usize).min(max_heads_per_threadgroup).max(1);

        let num_head_tiles = (head_count as usize).div_ceil(heads_per_threadgroup);

        let threadgroups_per_grid = MTLSize {
            width: args.batch_size as usize,
            height: num_head_tiles,
            depth: 1,
        };

        let threads_per_threadgroup = MTLSize {
            width: heads_per_threadgroup * simd_width,
            height: 1,
            depth: 1,
        };

        compute_encoder.dispatch_threadgroups(threadgroups_per_grid, threads_per_threadgroup);

        Ok(())
    }
}
