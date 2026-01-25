use crate::{
    DataType,
    backends::metal::{
        ComputeEncoderSetValue, MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState,
        MTLContext, MTLSize, ProtocolObject, Retained, encodable_block::EmbeddingError,
    },
    config::QuantizationMode,
};

// ---- Full Precision Embedding Lookup Kernel ----

pub struct FullPrecisionEmbeddingLookupKernel {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

#[derive(Debug)]
pub struct FullPrecisionEmbeddingLookupArguments<'a> {
    pub token_ids_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [batch_size] as U64
    pub weights_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [vocab_size, model_dim]
    pub output_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [batch_size, model_dim]
    pub batch_size: u32,
    pub vocab_size: u32,
    pub model_dim: u32,
    pub input_scale: f32,
}

impl FullPrecisionEmbeddingLookupKernel {
    pub fn new(
        mtl_context: &MTLContext,
        data_type: DataType,
    ) -> Result<Self, EmbeddingError> {
        let dtype_suffix = match data_type {
            DataType::F32 => "f32",
            DataType::F16 => "f16",
            DataType::BF16 => "bf16",
            other => {
                return Err(EmbeddingError::UnsupportedDataType(other));
            },
        };
        let kernel_name = format!("full_precision_embedding_lookup_{}", dtype_suffix);

        let pipeline = mtl_context
            .compute_pipeline_state(&kernel_name, None)
            .map_err(EmbeddingError::MetalError)?;

        Ok(Self { pipeline })
    }

    pub fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: FullPrecisionEmbeddingLookupArguments,
    ) -> Result<(), EmbeddingError> {
        encoder.set_compute_pipeline_state(&self.pipeline);

        MTLComputeCommandEncoder::set_buffer(
            encoder,
            Some(args.token_ids_buffer),
            0,
            0,
        );
        MTLComputeCommandEncoder::set_buffer(
            encoder,
            Some(args.weights_buffer),
            0,
            1,
        );
        MTLComputeCommandEncoder::set_buffer(
            encoder,
            Some(args.output_buffer),
            0,
            2,
        );

        encoder.set_value(&args.batch_size, 3);
        encoder.set_value(&args.vocab_size, 4);
        encoder.set_value(&args.model_dim, 5);
        encoder.set_value(&args.input_scale, 6);

        let total_threads = (args.batch_size * args.model_dim) as u64;
        let threads_per_threadgroup = 256u64;
        let threadgroups = (total_threads + threads_per_threadgroup - 1)
            / threads_per_threadgroup;

        MTLComputeCommandEncoder::dispatch_threadgroups(
            encoder,
            MTLSize::new(threadgroups as usize, 1, 1),
            MTLSize::new(threads_per_threadgroup as usize, 1, 1),
        );

        Ok(())
    }
}

// ---- Quantized Embedding Lookup Kernel ----

pub struct QuantizedEmbeddingLookupKernel {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

#[derive(Debug)]
pub struct QuantizedEmbeddingLookupArguments<'a> {
    pub token_ids_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [batch_size] as U64
    pub weights_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [vocab_size, model_dim/packing_divisor] as U8/I8
    pub scales_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [vocab_size, num_groups]
    pub biases_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [vocab_size, num_groups]
    pub output_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [batch_size, model_dim]
    pub batch_size: u32,
    pub vocab_size: u32,
    pub model_dim: u32,
    pub group_size: u32,
    pub input_scale: f32,
}

impl QuantizedEmbeddingLookupKernel {
    pub fn new(
        mtl_context: &MTLContext,
        data_type: DataType,
        mode: QuantizationMode,
    ) -> Result<Self, EmbeddingError> {
        let dtype_suffix = match data_type {
            DataType::F32 => "f32",
            DataType::F16 => "f16",
            DataType::BF16 => "bf16",
            other => {
                return Err(EmbeddingError::UnsupportedDataType(other));
            },
        };
        let mode_suffix = match mode {
            QuantizationMode::UInt4 => "uint4",
            QuantizationMode::Int8 => "int8",
            QuantizationMode::UInt8 => "uint8",
        };
        let kernel_name = format!(
            "quantized_embedding_lookup_{}_{}",
            dtype_suffix, mode_suffix
        );

        let pipeline = mtl_context
            .compute_pipeline_state(&kernel_name, None)
            .map_err(EmbeddingError::MetalError)?;

        Ok(Self { pipeline })
    }

    pub fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: QuantizedEmbeddingLookupArguments,
    ) -> Result<(), EmbeddingError> {
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set buffers
        encoder.set_buffer(Some(args.token_ids_buffer), 0, 0);
        encoder.set_buffer(Some(args.weights_buffer), 0, 1);
        encoder.set_buffer(Some(args.scales_buffer), 0, 2);
        encoder.set_buffer(Some(args.biases_buffer), 0, 3);
        encoder.set_buffer(Some(args.output_buffer), 0, 4);

        // Set constants
        encoder.set_value(&args.batch_size, 5);
        encoder.set_value(&args.vocab_size, 6);
        encoder.set_value(&args.model_dim, 7);
        encoder.set_value(&args.group_size, 8);
        encoder.set_value(&args.input_scale, 9);

        // Dispatch one thread per output element
        let total_threads = (args.batch_size * args.model_dim) as u64;
        let threads_per_threadgroup = 256u64;
        let threadgroups = (total_threads + threads_per_threadgroup - 1)
            / threads_per_threadgroup;

        encoder.dispatch_threadgroups(
            MTLSize::new(threadgroups as usize, 1, 1),
            MTLSize::new(threads_per_threadgroup as usize, 1, 1),
        );

        Ok(())
    }
}
