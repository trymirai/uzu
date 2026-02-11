//! QK Normalization encodable.

use super::super::{EncodableBlock, Metal};
use crate::{
    DataType,
    backends::{
        common::kernel::QKNormKernel,
        metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLContext, MTLDeviceExt,
            MTLError, MTLResourceOptions, ProtocolObject, Retained, kernel::dsl::QKNormMetalKernel,
        },
    },
    config::{NormalizationConfig, UpcastMode},
    encodable_block::EncodingParameters,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::ParameterTree,
};

pub struct QKNorm {
    query_kernel: Option<QKNormMetalKernel>,
    key_kernel: Option<QKNormMetalKernel>,
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
        parameter_tree: &ParameterTree<MTLContext>,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, MTLError> {
        let mut query_kernel = None;
        let mut key_kernel = None;
        let mut query_scales_buffer = None;
        let mut key_scales_buffer = None;

        // Setup query normalization if configured
        if let Some(ref q_config) = query_config {
            let scales_param = parameter_tree.leaf("query_norm.scales").map_err(|e| {
                MTLError::Library(crate::backends::metal::error::LibraryError::Custom(format!(
                    "Failed to load query scales: {:?}",
                    e
                )))
            })?;

            let scales_data = scales_param.as_bytes();
            let scales_buffer =
                context.device.new_buffer_with_data(scales_data, MTLResourceOptions::STORAGE_MODE_SHARED);

            let accumulation_data_type: DataType = q_config.accumulation_precision.into();
            let scale_data_type: DataType = q_config.scale_precision.into();

            let (input_type, scales_type, output_type) = match q_config.upcast_mode {
                UpcastMode::OnlyNormalization => (intermediate_data_type, scale_data_type, scale_data_type),
                UpcastMode::FullLayer => (intermediate_data_type, scale_data_type, scale_data_type),
            };

            let kernel = QKNormMetalKernel::new(context, input_type, scales_type, output_type, accumulation_data_type)?;

            query_kernel = Some(kernel);
            query_scales_buffer = Some(scales_buffer);
        }

        // Setup key normalization if configured
        if let Some(ref k_config) = key_config {
            let scales_param = parameter_tree.leaf("key_norm.scales").map_err(|e| {
                MTLError::Library(crate::backends::metal::error::LibraryError::Custom(format!(
                    "Failed to load key scales: {:?}",
                    e
                )))
            })?;

            let scales_data = scales_param.as_bytes();
            let scales_buffer =
                context.device.new_buffer_with_data(scales_data, MTLResourceOptions::STORAGE_MODE_SHARED);

            let accumulation_data_type: DataType = k_config.accumulation_precision.into();
            let scale_data_type: DataType = k_config.scale_precision.into();

            let (input_type, scales_type, output_type) = match k_config.upcast_mode {
                UpcastMode::OnlyNormalization => (intermediate_data_type, scale_data_type, scale_data_type),
                UpcastMode::FullLayer => (intermediate_data_type, scale_data_type, scale_data_type),
            };

            let kernel = QKNormMetalKernel::new(context, input_type, scales_type, output_type, accumulation_data_type)?;

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
        state: &mut ForwardPassState<Metal>,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        parameters: &EncodingParameters<Metal>,
    ) {
        let compute_encoder =
            command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
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
        state: &mut ForwardPassState<Metal>,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        _parameters: &EncodingParameters<Metal>,
    ) {
        let qkv_binding = state.arrays(&[self.qkv_array_id]);
        let qkv_shape = {
            let qkv_array = qkv_binding[0].borrow();
            qkv_array.shape().to_vec()
        };

        let qkv_array = qkv_binding[0].borrow_mut();
        let qkv_buffer = qkv_array.buffer();

        let batch_size = qkv_shape[0] as i32;
        let head_dim = self.head_dim as i32;

        // Process query normalization if configured
        if let (Some(query_kernel), Some(query_scales_buffer), Some(query_config)) =
            (&self.query_kernel, &self.query_scales_buffer, &self.query_config)
        {
            let query_head_count = self.num_q_heads as u32;
            query_kernel.encode(
                qkv_buffer,
                query_scales_buffer,
                qkv_buffer,
                batch_size as u32,
                self.num_q_heads as u32,
                self.num_kv_heads as u32,
                head_dim as u32,
                query_config.epsilon,
                query_config.scale_offset.unwrap_or(0.0),
                0,
                query_head_count,
                query_config.upcast_mode == UpcastMode::FullLayer,
                compute_encoder,
            );
        }

        // Process key normalization if configured
        if let (Some(key_kernel), Some(key_scales_buffer), Some(key_config)) =
            (&self.key_kernel, &self.key_scales_buffer, &self.key_config)
        {
            let key_head_count = self.num_kv_heads as u32;
            key_kernel.encode(
                qkv_buffer,
                key_scales_buffer,
                qkv_buffer,
                batch_size as u32,
                self.num_q_heads as u32,
                self.num_kv_heads as u32,
                head_dim as u32,
                key_config.epsilon,
                key_config.scale_offset.unwrap_or(0.0),
                self.num_q_heads as u32,
                key_head_count,
                key_config.upcast_mode == UpcastMode::FullLayer,
                compute_encoder,
            );
        }
    }
}
