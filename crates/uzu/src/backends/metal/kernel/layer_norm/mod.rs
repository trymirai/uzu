use std::{ffi::c_void, mem::size_of, ptr::NonNull};

use metal::MTLComputeCommandEncoder;

use crate::{
    DataType,
    backends::metal::{
        MTLBuffer, MTLComputePipelineState, MTLContext, MTLError, MTLSize,
        ProtocolObject, Retained,
    },
};

#[derive(Debug)]
pub struct LayerNormArguments<'a> {
    pub input_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub scales_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub output_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub batch_size: i32,
    pub model_dim: i32,
    pub epsilon: f32,
    pub scale_offset: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum LayerNormError {
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
}

pub struct LayerNormKernel {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl LayerNormKernel {
    pub fn new(
        context: &MTLContext,
        input_data_type: DataType,
        scales_data_type: DataType,
        output_data_type: DataType,
        accumulation_data_type: DataType,
        full_layer: bool,
    ) -> Result<Self, LayerNormError> {
        let kernel_name = Self::kernel_name(
            input_data_type,
            scales_data_type,
            output_data_type,
            accumulation_data_type,
            full_layer,
        )?;

        let pipeline = context
            .compute_pipeline_state(&kernel_name, None)
            .map_err(LayerNormError::MetalError)?;

        Ok(Self {
            pipeline,
        })
    }

    fn kernel_name(
        input: DataType,
        scales: DataType,
        output: DataType,
        accumulation: DataType,
        full_layer: bool,
    ) -> Result<String, LayerNormError> {
        let input_str = Self::type_to_string(input)?;
        let scales_str = Self::type_to_string(scales)?;
        let output_str = Self::type_to_string(output)?;
        let accum_str = Self::type_to_string(accumulation)?;
        let mode_str = if full_layer {
            "full"
        } else {
            "norm"
        };

        Ok(format!(
            "layer_norm_{}_{}_{}_{}_{}",
            input_str, scales_str, output_str, accum_str, mode_str
        ))
    }

    fn type_to_string(
        data_type: DataType
    ) -> Result<&'static str, LayerNormError> {
        match data_type {
            DataType::F32 => Ok("f32"),
            DataType::F16 => Ok("f16"),
            DataType::BF16 => Ok("bf16"),
            _ => Err(LayerNormError::UnsupportedDataType {
                input: data_type,
                scales: data_type,
                output: data_type,
                accumulation: data_type,
            }),
        }
    }

    pub fn encode(
        &self,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: LayerNormArguments,
    ) -> Result<(), LayerNormError> {
        compute_encoder.set_compute_pipeline_state(&self.pipeline);
        compute_encoder.set_buffer(Some(args.input_buffer), 0, 0);
        compute_encoder.set_buffer(Some(args.scales_buffer), 0, 1);
        compute_encoder.set_buffer(Some(args.output_buffer), 0, 2);
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&args.batch_size as *const i32 as *mut c_void)
                    .unwrap(),
                size_of::<i32>(),
                3,
            );
            compute_encoder.set_bytes(
                NonNull::new(&args.model_dim as *const i32 as *mut c_void)
                    .unwrap(),
                size_of::<i32>(),
                4,
            );
            compute_encoder.set_bytes(
                NonNull::new(&args.epsilon as *const f32 as *mut c_void)
                    .unwrap(),
                size_of::<f32>(),
                5,
            );
            compute_encoder.set_bytes(
                NonNull::new(&args.scale_offset as *const f32 as *mut c_void)
                    .unwrap(),
                size_of::<f32>(),
                6,
            );
        }

        let threadgroups_per_grid = MTLSize {
            width: args.batch_size as usize,
            height: 1,
            depth: 1,
        };

        let threads_per_threadgroup = MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        };

        compute_encoder.dispatch_threadgroups(
            threadgroups_per_grid,
            threads_per_threadgroup,
        );

        Ok(())
    }
}
