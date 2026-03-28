//! QK Normalization encodable.

use std::{
    cell::RefCell,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use thiserror::Error;

use crate::{
    DataType,
    backends::common::{
        Backend, Encoder,
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
    ParameterError(ParameterLoaderError<B>),
}

pub struct QKNorm<B: Backend> {
    query_kernel: Option<<B::Kernels as Kernels>::QKNormKernel>,
    key_kernel: Option<<B::Kernels as Kernels>::QKNormKernel>,
    query_config: Option<NormalizationConfig>,
    key_config: Option<NormalizationConfig>,
    qkv_array_id: ArrayId,
    query_scales_buffer: Option<Rc<RefCell<B::Buffer>>>,
    key_scales_buffer: Option<Rc<RefCell<B::Buffer>>>,
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
            let scales = parameter_tree.leaf_array("query_norm.scales").map_err(QKNormError::ParameterError)?;

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
            query_scales_buffer = Some(scales.buffer());
        }

        // Setup key normalization if configured
        if let Some(ref k_config) = key_config {
            let scales = parameter_tree.leaf_array("key_norm.scales").map_err(QKNormError::ParameterError)?;

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
            key_scales_buffer = Some(scales.buffer());
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

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let qkv_array = state.array(self.qkv_array_id);
        let batch_size = qkv_array.shape()[0] as u32;

        // Process query normalization if configured
        if let (Some(query_kernel), Some(query_scales_buffer), Some(query_config)) =
            (&self.query_kernel, &self.query_scales_buffer, &self.query_config)
        {
            query_kernel.encode(
                None::<&B::Buffer>,
                query_scales_buffer.borrow().deref(),
                qkv_array.buffer().borrow_mut().deref_mut(),
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
                None::<&B::Buffer>,
                key_scales_buffer.borrow().deref(),
                qkv_array.buffer().borrow_mut().deref_mut(),
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
        Ok(())
    }
}
