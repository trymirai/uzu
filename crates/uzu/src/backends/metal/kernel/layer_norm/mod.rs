use std::{mem::size_of, rc::Rc};

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};
use mpsgraph::CommandBuffer as MPSCommandBuffer;

use crate::{
    Array, DataType,
    backends::metal::{
        MTLContext, MTLError,
        forward_pass::{
            ArrayId, ForwardPassState,
            encodable_with_state::{EncodableWithState, EncodingParameters},
        },
    },
    config::{NormalizationConfig, UpcastMode},
    parameters::ParameterTree,
};

#[derive(Debug)]
pub struct LayerNormArguments<'a> {
    pub input_buffer: &'a MTLBuffer,
    pub scales_buffer: &'a MTLBuffer,
    pub output_buffer: &'a MTLBuffer,
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
    pipeline: MTLComputePipelineState,
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

        eprintln!(
            "[DEBUG] LayerNormKernel::new - Requesting kernel: {}",
            kernel_name
        );
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
        compute_encoder: &ComputeCommandEncoderRef,
        args: LayerNormArguments,
    ) -> Result<(), LayerNormError> {
        compute_encoder.set_compute_pipeline_state(&self.pipeline);
        compute_encoder.set_buffer(0, Some(args.input_buffer), 0);
        compute_encoder.set_buffer(1, Some(args.scales_buffer), 0);
        compute_encoder.set_buffer(2, Some(args.output_buffer), 0);
        compute_encoder.set_bytes(
            3,
            size_of::<i32>() as u64,
            &args.batch_size as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<i32>() as u64,
            &args.model_dim as *const i32 as *const _,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<f32>() as u64,
            &args.epsilon as *const f32 as *const _,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<f32>() as u64,
            &args.scale_offset as *const f32 as *const _,
        );

        let threadgroups_per_grid = MTLSize {
            width: args.batch_size as u64,
            height: 1,
            depth: 1,
        };

        let threads_per_threadgroup = MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        };

        compute_encoder.dispatch_thread_groups(
            threadgroups_per_grid,
            threads_per_threadgroup,
        );

        Ok(())
    }
}

pub struct LayerNormKernelEncodable {
    kernel: LayerNormKernel,
    config: NormalizationConfig,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
    scales_buffer: MTLBuffer,
}

impl LayerNormKernelEncodable {
    pub fn new(
        context: &MTLContext,
        intermediate_data_type: DataType,
        config: NormalizationConfig,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) -> Result<Self, LayerNormError> {
        // Load scales from parameter tree
        let scales_param = parameter_tree.leaf("scales").map_err(|e| {
            LayerNormError::MetalError(MTLError::Library(
                crate::backends::metal::error::LibraryError::Custom(format!(
                    "Failed to load scales: {:?}",
                    e
                )),
            ))
        })?;

        let scales_data = scales_param.buffer();
        let scales_buffer = context.device.new_buffer_with_data(
            scales_data.as_ptr() as *const _,
            scales_data.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let accumulation_data_type: DataType =
            config.accumulation_precision.into();
        let scale_data_type: DataType = config.scale_precision.into();

        let (input_type, scales_type, output_type) = match config.upcast_mode {
            UpcastMode::OnlyNormalization => {
                (intermediate_data_type, scale_data_type, scale_data_type)
            },
            UpcastMode::FullLayer => {
                (intermediate_data_type, scale_data_type, scale_data_type)
            },
        };

        let kernel = LayerNormKernel::new(
            context,
            input_type,
            scales_type,
            output_type,
            accumulation_data_type,
            config.upcast_mode == UpcastMode::FullLayer,
        )?;

        Ok(Self {
            kernel,
            config,
            input_array_id,
            output_array_id,
            scales_buffer,
        })
    }
}

impl EncodableWithState for LayerNormKernelEncodable {
    fn encode(
        &self,
        state: &mut dyn ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        let input_binding = state.arrays(&[self.input_array_id]);
        let output_binding = state.arrays(&[self.output_array_id]);

        let input_shape = {
            let input_array = input_binding[0].borrow();
            input_array.shape().to_vec()
        };

        let mut input_array = input_binding[0].borrow_mut();
        let mut output_array = output_binding[0].borrow_mut();

        let input_buffer = unsafe { input_array.mtl_buffer() };
        let output_buffer = unsafe { output_array.mtl_buffer() };

        let batch_size = input_shape[0] as i32;
        let model_dim = input_shape[1] as i32;

        let mtl_command_buffer =
            command_buffer.root_command_buffer().to_owned();
        let compute_encoder = mtl_command_buffer.new_compute_command_encoder();

        if let Err(e) = self.kernel.encode(
            &compute_encoder,
            LayerNormArguments {
                input_buffer: &input_buffer,
                scales_buffer: &self.scales_buffer,
                output_buffer: &output_buffer,
                batch_size,
                model_dim,
                epsilon: self.config.epsilon,
                scale_offset: self.config.scale_offset.unwrap_or(0.0),
            },
        ) {
            eprintln!("Failed to encode LayerNorm kernel: {:?}", e);
        }

        compute_encoder.end_encoding();

        if parameters.wait_until_completed {
            command_buffer.commit_and_continue();
            mtl_command_buffer.wait_until_completed();
        }
    }
}
