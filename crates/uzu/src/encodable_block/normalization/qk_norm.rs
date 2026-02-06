//! QK Normalization encodable.

use std::rc::Rc;

use super::super::{EncodableBlock, EncodingParameters, Metal};
use crate::{
    Array, DataType,
    backends::metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
        MTLComputeCommandEncoder, MTLContext, MTLDeviceExt, MTLError,
        MTLResourceOptions, ProtocolObject, Retained,
        forward_pass::{ArrayId, ForwardPassState},
        kernel::rms_norm::{
            QKNormArguments, QKNormTarget, RMSNormError, RMSNormKernel,
            RMSNormKernelType,
        },
    },
    config::{NormalizationConfig, UpcastMode},
    parameters::ParameterTree,
};

pub struct QKNorm {
    query_kernel: Option<RMSNormKernel>,
    key_kernel: Option<RMSNormKernel>,
    query_config: Option<NormalizationConfig>,
    key_config: Option<NormalizationConfig>,
    qkv_array_id: ArrayId,
    query_scales_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    key_scales_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl QKNorm {
    pub fn new(
        context: &MTLContext,
        intermediate_data_type: DataType,
        query_config: Option<NormalizationConfig>,
        key_config: Option<NormalizationConfig>,
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
                scales_data,
                MTLResourceOptions::STORAGE_MODE_SHARED,
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
                scales_data,
                MTLResourceOptions::STORAGE_MODE_SHARED,
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
            query_scales_buffer: query_scales_buffer.flatten(),
            key_scales_buffer: key_scales_buffer.flatten(),
            num_q_heads,
            num_kv_heads,
            head_dim,
        })
    }
}

impl EncodableBlock<Metal> for QKNorm {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        parameters: &EncodingParameters,
    ) {
        let compute_encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        self.encode_with_shared_encoder(state, &compute_encoder, parameters);
        compute_encoder.end_encoding();

        if parameters.wait_until_completed {
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        _parameters: &EncodingParameters,
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

        // Process query normalization if configured
        if let (
            Some(query_kernel),
            Some(query_scales_buffer),
            Some(query_config),
        ) =
            (&self.query_kernel, &self.query_scales_buffer, &self.query_config)
        {
            if let Err(e) = query_kernel.encode_qk_norm(
                compute_encoder,
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
                compute_encoder,
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
    }
}
