use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use super::super::MTLContext;
use crate::{
    DataType, backends::metal::encodable_block::EmbeddingError,
    config::QuantizationMode,
};

// ---- Full Precision Embedding Lookup Kernel ----

pub struct FullPrecisionEmbeddingLookupKernel {
    pipeline: MTLComputePipelineState,
}

#[derive(Debug)]
pub struct FullPrecisionEmbeddingLookupArguments<'a> {
    pub token_ids_buffer: &'a MTLBuffer, // [batch_size] as U64
    pub weights_buffer: &'a MTLBuffer,   // [vocab_size, model_dim]
    pub output_buffer: &'a MTLBuffer,    // [batch_size, model_dim]
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
                return Err(EmbeddingError::UnsupportedDataType(
                    other,
                ));
            },
        };
        let kernel_name =
            format!("full_precision_embedding_lookup_{}", dtype_suffix);

        let (pipeline, _) = mtl_context
            .compute_pipeline_state_with_reflection(&kernel_name, None)
            .map_err(EmbeddingError::MetalError)?;

        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        args: FullPrecisionEmbeddingLookupArguments,
    ) -> Result<(), EmbeddingError> {
        encoder.set_compute_pipeline_state(&self.pipeline);

        encoder.set_buffer(0, Some(args.token_ids_buffer), 0);
        encoder.set_buffer(1, Some(args.weights_buffer), 0);
        encoder.set_buffer(2, Some(args.output_buffer), 0);

        encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &args.batch_size as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &args.vocab_size as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &args.model_dim as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            size_of::<f32>() as u64,
            &args.input_scale as *const f32 as *const _,
        );

        let total_threads = (args.batch_size * args.model_dim) as u64;
        let threads_per_threadgroup = 256u64;
        let threadgroups = (total_threads + threads_per_threadgroup - 1)
            / threads_per_threadgroup;

        encoder.dispatch_thread_groups(
            MTLSize::new(threadgroups, 1, 1),
            MTLSize::new(threads_per_threadgroup, 1, 1),
        );

        Ok(())
    }
}

// ---- Quantized Embedding Lookup Kernel ----

pub struct QuantizedEmbeddingLookupKernel {
    pipeline: MTLComputePipelineState,
}

#[derive(Debug)]
pub struct QuantizedEmbeddingLookupArguments<'a> {
    pub token_ids_buffer: &'a MTLBuffer, // [batch_size] as U64
    pub weights_buffer: &'a MTLBuffer, // [vocab_size, model_dim/packing_divisor] as U8/I8
    pub scales_buffer: &'a MTLBuffer,  // [vocab_size, num_groups]
    pub biases_buffer: &'a MTLBuffer,  // [vocab_size, num_groups]
    pub output_buffer: &'a MTLBuffer,  // [batch_size, model_dim]
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
                return Err(EmbeddingError::UnsupportedDataType(
                    other,
                ));
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

        let (pipeline, _) = mtl_context
            .compute_pipeline_state_with_reflection(&kernel_name, None)
            .map_err(EmbeddingError::MetalError)?;

        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        args: QuantizedEmbeddingLookupArguments,
    ) -> Result<(), EmbeddingError> {
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set buffers
        encoder.set_buffer(0, Some(args.token_ids_buffer), 0);
        encoder.set_buffer(1, Some(args.weights_buffer), 0);
        encoder.set_buffer(2, Some(args.scales_buffer), 0);
        encoder.set_buffer(3, Some(args.biases_buffer), 0);
        encoder.set_buffer(4, Some(args.output_buffer), 0);

        // Set constants
        encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &args.batch_size as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &args.vocab_size as *const u32 as *const _,
        );
        encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            &args.model_dim as *const u32 as *const _,
        );
        encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            &args.group_size as *const u32 as *const _,
        );
        encoder.set_bytes(
            9,
            size_of::<f32>() as u64,
            &args.input_scale as *const f32 as *const _,
        );

        // Dispatch one thread per output element
        let total_threads = (args.batch_size * args.model_dim) as u64;
        let threads_per_threadgroup = 256u64;
        let threadgroups = (total_threads + threads_per_threadgroup - 1)
            / threads_per_threadgroup;

        encoder.dispatch_thread_groups(
            MTLSize::new(threadgroups, 1, 1),
            MTLSize::new(threads_per_threadgroup, 1, 1),
        );

        Ok(())
    }
}
