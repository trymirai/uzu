//! QK Normalization encodable.

use thiserror::Error;

use super::{EncodableBlock, EncodingParameters};
use crate::{
    DataType,
    backends::common::{
        Backend,
        kernel::{Kernels, QKNormKernel},
    },
    config::{NormalizationConfig, UpcastMode},
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum QKNormError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(ParameterLoaderError),
}

pub struct QKNorm<B: Backend> {
    query_kernel: Option<<B::Kernels as Kernels>::QKNormKernel>,
    key_kernel: Option<<B::Kernels as Kernels>::QKNormKernel>,
    query_config: Option<NormalizationConfig>,
    key_config: Option<NormalizationConfig>,
    qkv_array_id: ArrayId,
    query_scales_buffer: Option<B::NativeBuffer>,
    key_scales_buffer: Option<B::NativeBuffer>,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl<B: Backend> QKNorm<B> {
    pub fn new(
        context: &B::Context,
        intermediate_data_type: DataType,
        query_config: Option<NormalizationConfig>,
        key_config: Option<NormalizationConfig>,
        qkv_array_id: ArrayId,
        parameter_tree: &ParameterTree<B::Context>,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, QKNormError<B>> {
        let mut query_kernel = None;
        let mut key_kernel = None;
        let mut query_scales_buffer = None;
        let mut key_scales_buffer = None;

        // Setup query normalization if configured
        if let Some(ref q_config) = query_config {
            let scales = parameter_tree
                .leaf("query_norm.scales")
                .map_err(QKNormError::ParameterError)?;
            let scales_buffer = scales.buffer().clone();

            let accumulation_data_type: DataType = q_config.accumulation_precision.into();
            let scale_data_type: DataType = q_config.scale_precision.into();
            let (input_type, scales_type, output_type) = match q_config.upcast_mode {
                UpcastMode::OnlyNormalization => (intermediate_data_type, scale_data_type, scale_data_type),
                UpcastMode::FullLayer => (intermediate_data_type, scale_data_type, scale_data_type),
            };

            let kernel = <B::Kernels as Kernels>::QKNormKernel::new(
                context,
                input_type,
                scales_type,
                output_type,
                accumulation_data_type,
            )
            .map_err(QKNormError::BackendError)?;

            query_kernel = Some(kernel);
            query_scales_buffer = Some(scales_buffer);
        }

        // Setup key normalization if configured
        if let Some(ref k_config) = key_config {
            let scales = parameter_tree
                .leaf("key_norm.scales")
                .map_err(QKNormError::ParameterError)?;
            let scales_buffer = scales.buffer().clone();

            let accumulation_data_type: DataType = k_config.accumulation_precision.into();
            let scale_data_type: DataType = k_config.scale_precision.into();
            let (input_type, scales_type, output_type) = match k_config.upcast_mode {
                UpcastMode::OnlyNormalization => (intermediate_data_type, scale_data_type, scale_data_type),
                UpcastMode::FullLayer => (intermediate_data_type, scale_data_type, scale_data_type),
            };

            let kernel = <B::Kernels as Kernels>::QKNormKernel::new(
                context,
                input_type,
                scales_type,
                output_type,
                accumulation_data_type,
            )
            .map_err(QKNormError::BackendError)?;

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

impl<B: Backend> EncodableBlock<B> for QKNorm<B> {
    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<B>,
        _parameters: &EncodingParameters<B>,
        encoder: &B::ComputeEncoder,
    ) {
        let qkv_binding = state.arrays(&[self.qkv_array_id]);
        let qkv_shape = {
            let qkv_array = qkv_binding[0].borrow();
            qkv_array.shape().to_vec()
        };

        let qkv_array = qkv_binding[0].borrow_mut();
        let qkv_buffer = qkv_array.buffer();
        let batch_size = qkv_shape[0] as u32;

        // Process query normalization if configured
        if let (Some(query_kernel), Some(query_scales_buffer), Some(query_config)) =
            (&self.query_kernel, &self.query_scales_buffer, &self.query_config)
        {
            query_kernel.encode(
                qkv_buffer,
                query_scales_buffer,
                qkv_buffer,
                batch_size,
                self.num_q_heads as u32,
                self.num_kv_heads as u32,
                self.head_dim as u32,
                query_config.epsilon,
                query_config.scale_offset.unwrap_or(0.0),
                0,
                self.num_q_heads as u32,
                query_config.upcast_mode == UpcastMode::FullLayer,
                encoder,
            );
        }

        // Process key normalization if configured
        if let (Some(key_kernel), Some(key_scales_buffer), Some(key_config)) =
            (&self.key_kernel, &self.key_scales_buffer, &self.key_config)
        {
            key_kernel.encode(
                qkv_buffer,
                key_scales_buffer,
                qkv_buffer,
                batch_size,
                self.num_q_heads as u32,
                self.num_kv_heads as u32,
                self.head_dim as u32,
                key_config.epsilon,
                key_config.scale_offset.unwrap_or(0.0),
                self.num_q_heads as u32,
                self.num_kv_heads as u32,
                key_config.upcast_mode == UpcastMode::FullLayer,
                encoder,
            );
        }
    }
}
