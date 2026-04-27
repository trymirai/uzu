//! QK Normalization encodable.

use thiserror::Error;

use crate::{
    DataType,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{Kernels, QKNormKernel},
    },
    config::{NormalizationConfig, UpcastMode},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum QKNormError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(ParameterLoaderError<B>),
}

pub struct QKNorm<B: Backend> {
    query_kernel: Option<<B::Kernels as Kernels>::QKNormKernel>,
    key_kernel: Option<<B::Kernels as Kernels>::QKNormKernel>,
    query_config: Option<NormalizationConfig>,
    key_config: Option<NormalizationConfig>,
    query_scales: Option<Allocation<B>>,
    key_scales: Option<Allocation<B>>,
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
        parameter_tree: &ParameterTree<B::Context>,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, QKNormError<B>> {
        let mut query_kernel = None;
        let mut key_kernel = None;
        let mut query_scales = None;
        let mut key_scales = None;

        // Setup query normalization if configured
        if let Some(ref q_config) = query_config {
            let scales = parameter_tree
                .leaf("query_norm.scales")
                .map_err(QKNormError::ParameterError)?
                .read_allocation()
                .map_err(QKNormError::ParameterError)?;

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
                true,
            )
            .map_err(QKNormError::BackendError)?;

            query_kernel = Some(kernel);
            query_scales = Some(scales);
        }

        // Setup key normalization if configured
        if let Some(ref k_config) = key_config {
            let scales = parameter_tree
                .leaf("key_norm.scales")
                .map_err(QKNormError::ParameterError)?
                .read_allocation()
                .map_err(QKNormError::ParameterError)?;

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
                true,
            )
            .map_err(QKNormError::BackendError)?;

            key_kernel = Some(kernel);
            key_scales = Some(scales);
        }

        Ok(Self {
            query_kernel,
            key_kernel,
            query_config,
            key_config,
            query_scales,
            key_scales,
            num_q_heads,
            num_kv_heads,
            head_dim,
        })
    }

    pub fn encode(
        &self,
        qkv: &mut Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        // Process query normalization if configured
        if let (Some(query_kernel), Some(query_scales), Some(query_config)) =
            (&self.query_kernel, &self.query_scales, &self.query_config)
        {
            query_kernel.encode(
                None::<&Allocation<B>>,
                query_scales,
                &mut *qkv,
                batch_dim as u32,
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
        if let (Some(key_kernel), Some(key_scales), Some(key_config)) =
            (&self.key_kernel, &self.key_scales, &self.key_config)
        {
            key_kernel.encode(
                None::<&Allocation<B>>,
                key_scales,
                &mut *qkv,
                batch_dim as u32,
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
        Ok(())
    }
}
