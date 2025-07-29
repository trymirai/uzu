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
    config::{RMSNormConfig, UpcastMode},
    parameters::ParameterTree,
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

impl QKNormTarget {
    fn to_mask(self) -> u32 {
        match self {
            QKNormTarget::QueryHeads => 0b01,
            QKNormTarget::KeyHeads => 0b10,
            QKNormTarget::Both => 0b11,
        }
    }
}

pub struct RMSNormKernel {
    pipeline: MTLComputePipelineState,
    kernel_type: RMSNormKernelType,
}

#[derive(Debug)]
pub struct RMSNormArguments<'a> {
    pub input_buffer: &'a MTLBuffer,
    pub scales_buffer: &'a MTLBuffer,
    pub output_buffer: &'a MTLBuffer,
    pub batch_size: i32,
    pub model_dim: i32,
    pub epsilon: f32,
    pub scale_offset: f32,
}

#[derive(Debug)]
pub struct QKNormArguments<'a> {
    pub qkv_input_buffer: &'a MTLBuffer,
    pub scales_buffer: &'a MTLBuffer,
    pub qkv_output_buffer: &'a MTLBuffer,
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

        let pipeline = context
            .compute_pipeline_state(&function_name, None)
            .map_err(RMSNormError::MetalError)?;

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
            base_name,
            input_suffix,
            scales_suffix,
            output_suffix,
            accum_suffix,
            mode_suffix
        ))
    }

    fn data_type_to_suffix(
        data_type: DataType
    ) -> Result<&'static str, RMSNormError> {
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

    fn accum_type_to_suffix(
        data_type: DataType
    ) -> Result<&'static str, RMSNormError> {
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

    pub fn encode(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: RMSNormArguments,
    ) -> Result<(), RMSNormError> {
        if self.kernel_type != RMSNormKernelType::Standard {
            return Err(RMSNormError::InvalidKernelType);
        }

        compute_encoder.set_compute_pipeline_state(&self.pipeline);

        // Set buffers
        compute_encoder.set_buffer(0, Some(args.input_buffer), 0);
        compute_encoder.set_buffer(1, Some(args.scales_buffer), 0);
        compute_encoder.set_buffer(2, Some(args.output_buffer), 0);

        // Set parameters
        compute_encoder.set_bytes(
            3,
            size_of::<i32>() as u64,
            &args.batch_size as *const i32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<i32>() as u64,
            &args.model_dim as *const i32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<f32>() as u64,
            &args.epsilon as *const f32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<f32>() as u64,
            &args.scale_offset as *const f32 as *const std::ffi::c_void,
        );

        let threads_per_threadgroup = MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        };

        let threadgroups_per_grid = MTLSize {
            width: args.batch_size as u64,
            height: 1,
            depth: 1,
        };

        compute_encoder.dispatch_thread_groups(
            threadgroups_per_grid,
            threads_per_threadgroup,
        );

        Ok(())
    }

    pub fn encode_qk_norm(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: QKNormArguments,
    ) -> Result<(), RMSNormError> {
        if self.kernel_type != RMSNormKernelType::QueryKey {
            return Err(RMSNormError::InvalidKernelType);
        }

        compute_encoder.set_compute_pipeline_state(&self.pipeline);

        // Set buffers
        compute_encoder.set_buffer(0, Some(args.qkv_input_buffer), 0);
        compute_encoder.set_buffer(1, Some(args.scales_buffer), 0);
        compute_encoder.set_buffer(2, Some(args.qkv_output_buffer), 0);

        // Set parameters
        compute_encoder.set_bytes(
            3,
            size_of::<i32>() as u64,
            &args.batch_size as *const i32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<i32>() as u64,
            &args.num_q_heads as *const i32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<i32>() as u64,
            &args.num_kv_heads as *const i32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<i32>() as u64,
            &args.head_dim as *const i32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            7,
            size_of::<f32>() as u64,
            &args.epsilon as *const f32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            8,
            size_of::<f32>() as u64,
            &args.scale_offset as *const f32 as *const std::ffi::c_void,
        );

        let active_type_mask = args.target.to_mask();
        compute_encoder.set_bytes(
            9,
            size_of::<u32>() as u64,
            &active_type_mask as *const u32 as *const std::ffi::c_void,
        );

        // Calculate threadgroup dimensions based on target
        let total_heads_to_dispatch = match args.target {
            QKNormTarget::QueryHeads => {
                // Process Q heads only: dispatch for indices 0 to num_q_heads-1
                args.num_q_heads
            },
            QKNormTarget::KeyHeads => {
                // Process K heads only: dispatch for indices 0 to num_q_heads+num_kv_heads-1
                // (kernel will skip Q heads based on target enum)
                args.num_q_heads + args.num_kv_heads
            },
            QKNormTarget::Both => {
                // Process both Q and K heads: dispatch for indices 0 to num_q_heads+num_kv_heads-1
                args.num_q_heads + args.num_kv_heads
            },
        };

        let threadgroups_per_grid = MTLSize {
            width: args.batch_size as u64,
            height: total_heads_to_dispatch as u64,
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

pub struct RMSNormKernelEncodable {
    kernel: RMSNormKernel,
    config: RMSNormConfig,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
    scales_buffer: MTLBuffer,
}

impl RMSNormKernelEncodable {
    pub fn new(
        context: &MTLContext,
        intermediate_data_type: DataType,
        config: RMSNormConfig,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
    ) -> Result<Self, RMSNormError> {
        // Load scales from parameter tree
        let scales_param = parameter_tree.leaf("scales").map_err(|e| {
            RMSNormError::MetalError(MTLError::Library(
                crate::backends::metal::error::LibraryError::Custom(format!(
                    "Failed to load scales: {:?}",
                    e
                )),
            ))
        })?;

        // TODO: Don't create buffers dynamically, we need to use forward pass storage for thing like this
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
                // Input stays as pipeline type, scales stay scale precision, output is scale precision
                (intermediate_data_type, scale_data_type, scale_data_type)
            },
            UpcastMode::FullLayer => {
                // Input stays as pipeline type, scales stay in original precision (will be cast to AccumT inside kernel), output is scale precision
                (intermediate_data_type, scale_data_type, scale_data_type)
            },
        };

        let kernel = RMSNormKernel::new_with_mode(
            context,
            input_type,
            scales_type,
            output_type,
            accumulation_data_type,
            RMSNormKernelType::Standard,
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

impl EncodableWithState for RMSNormKernelEncodable {
    fn encode(
        &self,
        state: &mut ForwardPassState,
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
            RMSNormArguments {
                input_buffer: &input_buffer,
                scales_buffer: &self.scales_buffer,
                output_buffer: &output_buffer,
                batch_size,
                model_dim,
                epsilon: self.config.epsilon,
                scale_offset: self.config.scale_offset.unwrap_or(0.0),
            },
        ) {
            eprintln!("Failed to encode RMS norm kernel: {:?}", e);
        }

        compute_encoder.end_encoding();

        if parameters.wait_until_completed {
            command_buffer.commit_and_continue();
            mtl_command_buffer.wait_until_completed();
        }
    }
}

pub struct QKNormKernelEncodable {
    query_kernel: Option<RMSNormKernel>,
    key_kernel: Option<RMSNormKernel>,
    query_config: Option<RMSNormConfig>,
    key_config: Option<RMSNormConfig>,
    qkv_array_id: ArrayId,
    query_scales_buffer: Option<MTLBuffer>,
    key_scales_buffer: Option<MTLBuffer>,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl QKNormKernelEncodable {
    pub fn new(
        context: &MTLContext,
        intermediate_data_type: DataType,
        query_config: Option<RMSNormConfig>,
        key_config: Option<RMSNormConfig>,
        qkv_array_id: ArrayId,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, RMSNormError> {
        let mut query_kernel = None;
        let mut key_kernel = None;
        let mut query_scales_buffer = None;
        let mut key_scales_buffer = None;

        // Setup query normalization if configured
        if let Some(ref q_config) = query_config {
            let scales_param =
                parameter_tree.leaf("query_norm.scales").map_err(|e| {
                    RMSNormError::MetalError(MTLError::Library(
                        crate::backends::metal::error::LibraryError::Custom(
                            format!("Failed to load query scales: {:?}", e),
                        ),
                    ))
                })?;

            let scales_data = scales_param.buffer();
            let scales_buffer = context.device.new_buffer_with_data(
                scales_data.as_ptr() as *const _,
                scales_data.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            let accumulation_data_type: DataType =
                q_config.accumulation_precision.into();
            let scale_data_type: DataType = q_config.scale_precision.into();

            let (input_type, scales_type, output_type) = match q_config
                .upcast_mode
            {
                UpcastMode::OnlyNormalization => {
                    (intermediate_data_type, scale_data_type, scale_data_type)
                },
                UpcastMode::FullLayer => {
                    (intermediate_data_type, scale_data_type, scale_data_type)
                },
            };

            let kernel = RMSNormKernel::new_with_mode(
                context,
                input_type,
                scales_type,
                output_type,
                accumulation_data_type,
                RMSNormKernelType::QueryKey,
                q_config.upcast_mode == UpcastMode::FullLayer,
            )?;

            query_kernel = Some(kernel);
            query_scales_buffer = Some(scales_buffer);
        }

        // Setup key normalization if configured
        if let Some(ref k_config) = key_config {
            let scales_param =
                parameter_tree.leaf("key_norm.scales").map_err(|e| {
                    RMSNormError::MetalError(MTLError::Library(
                        crate::backends::metal::error::LibraryError::Custom(
                            format!("Failed to load key scales: {:?}", e),
                        ),
                    ))
                })?;

            let scales_data = scales_param.buffer();
            let scales_buffer = context.device.new_buffer_with_data(
                scales_data.as_ptr() as *const _,
                scales_data.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            let accumulation_data_type: DataType =
                k_config.accumulation_precision.into();
            let scale_data_type: DataType = k_config.scale_precision.into();

            let (input_type, scales_type, output_type) = match k_config
                .upcast_mode
            {
                UpcastMode::OnlyNormalization => {
                    (intermediate_data_type, scale_data_type, scale_data_type)
                },
                UpcastMode::FullLayer => {
                    (intermediate_data_type, scale_data_type, scale_data_type)
                },
            };

            let kernel = RMSNormKernel::new_with_mode(
                context,
                input_type,
                scales_type,
                output_type,
                accumulation_data_type,
                RMSNormKernelType::QueryKey,
                k_config.upcast_mode == UpcastMode::FullLayer,
            )?;

            key_kernel = Some(kernel);
            key_scales_buffer = Some(scales_buffer);
        }

        Ok(Self {
            query_kernel,
            key_kernel,
            query_config,
            key_config,
            qkv_array_id,
            query_scales_buffer,
            key_scales_buffer,
            num_q_heads,
            num_kv_heads,
            head_dim,
        })
    }
}

impl EncodableWithState for QKNormKernelEncodable {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        let qkv_binding = state.arrays(&[self.qkv_array_id]);
        let qkv_shape = {
            let qkv_array = qkv_binding[0].borrow();
            qkv_array.shape().to_vec()
        };

        let mut qkv_array = qkv_binding[0].borrow_mut();
        let qkv_buffer = unsafe { qkv_array.mtl_buffer() };

        let batch_size = qkv_shape[0] as i32;
        let head_dim = self.head_dim as i32;

        let mtl_command_buffer =
            command_buffer.root_command_buffer().to_owned();
        let compute_encoder = mtl_command_buffer.new_compute_command_encoder();

        // Process query normalization if configured
        if let (
            Some(query_kernel),
            Some(query_scales_buffer),
            Some(query_config),
        ) =
            (&self.query_kernel, &self.query_scales_buffer, &self.query_config)
        {
            if let Err(e) = query_kernel.encode_qk_norm(
                &compute_encoder,
                QKNormArguments {
                    qkv_input_buffer: &qkv_buffer,
                    scales_buffer: query_scales_buffer,
                    qkv_output_buffer: &qkv_buffer,
                    batch_size,
                    // Always pass actual head counts (needed for correct buffer addressing)
                    num_q_heads: self.num_q_heads as i32,
                    num_kv_heads: self.num_kv_heads as i32,
                    head_dim,
                    epsilon: query_config.epsilon,
                    scale_offset: query_config.scale_offset.unwrap_or(0.0),
                    target: QKNormTarget::QueryHeads,
                },
            ) {
                eprintln!(
                    "Failed to encode query normalization kernel: {:?}",
                    e
                );
            }
        }

        // Process key normalization if configured
        if let (Some(key_kernel), Some(key_scales_buffer), Some(key_config)) =
            (&self.key_kernel, &self.key_scales_buffer, &self.key_config)
        {
            if let Err(e) = key_kernel.encode_qk_norm(
                &compute_encoder,
                QKNormArguments {
                    qkv_input_buffer: &qkv_buffer,
                    scales_buffer: key_scales_buffer,
                    qkv_output_buffer: &qkv_buffer,
                    batch_size,
                    // Always pass actual head counts (needed for correct buffer addressing)
                    num_q_heads: self.num_q_heads as i32,
                    num_kv_heads: self.num_kv_heads as i32,
                    head_dim,
                    epsilon: key_config.epsilon,
                    scale_offset: key_config.scale_offset.unwrap_or(0.0),
                    target: QKNormTarget::KeyHeads,
                },
            ) {
                eprintln!("Failed to encode key normalization kernel: {:?}", e);
            }
        }

        compute_encoder.end_encoding();

        if parameters.wait_until_completed {
            command_buffer.commit_and_continue();
            mtl_command_buffer.wait_until_completed();
        }
    }
}
