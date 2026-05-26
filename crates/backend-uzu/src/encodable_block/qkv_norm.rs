use thiserror::Error;

use crate::{
    DataType,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{Kernels, QKVNormKernel},
    },
    config::normalization::{NormalizationConfig, UpcastMode},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum QKVNormError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
}

struct Head<B: Backend> {
    kernel: <B::Kernels as Kernels>::QKVNormKernel,
    scales: Option<Allocation<B>>,
    config: NormalizationConfig,
}

pub struct QKVNorm<B: Backend> {
    query: Option<Head<B>>,
    key: Option<Head<B>>,
    value: Option<Head<B>>,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl<B: Backend> QKVNorm<B> {
    pub fn new(
        context: &B::Context,
        intermediate_data_type: DataType,
        query_config: Option<NormalizationConfig>,
        key_config: Option<NormalizationConfig>,
        value_config: Option<NormalizationConfig>,
        parameter_tree: &ParameterTree<B>,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, QKVNormError<B>> {
        let query = query_config
            .map(|cfg| {
                Self::build_head(
                    context,
                    intermediate_data_type,
                    cfg,
                    parameter_tree,
                    Some("query_norm.scales"),
                    head_dim,
                )
            })
            .transpose()?;
        let key = key_config
            .map(|cfg| {
                Self::build_head(
                    context,
                    intermediate_data_type,
                    cfg,
                    parameter_tree,
                    Some("key_norm.scales"),
                    head_dim,
                )
            })
            .transpose()?;
        let value = value_config
            .map(|cfg| Self::build_head(context, intermediate_data_type, cfg, parameter_tree, None, head_dim))
            .transpose()?;

        Ok(Self {
            query,
            key,
            value,
            num_q_heads,
            num_kv_heads,
            head_dim,
        })
    }

    fn build_head(
        context: &B::Context,
        intermediate_data_type: DataType,
        config: NormalizationConfig,
        parameter_tree: &ParameterTree<B>,
        scales_leaf: Option<&str>,
        head_dim: usize,
    ) -> Result<Head<B>, QKVNormError<B>> {
        let (scales, scale_data_type) = if let Some(scales_leaf) = scales_leaf {
            let scale_data_type = super::normalization::NORMALIZATION_SCALE_DATA_TYPE;
            (
                Some(parameter_tree.leaf(scales_leaf)?.validate(&[head_dim], scale_data_type)?.read_allocation()?),
                scale_data_type,
            )
        } else {
            (None, intermediate_data_type)
        };
        let scale_free = scales.is_none();
        let accumulation_data_type = super::normalization::NORMALIZATION_ACCUMULATION_DATA_TYPE;
        let output_data_type = intermediate_data_type;
        let kernel = <B::Kernels as Kernels>::QKVNormKernel::new(
            context,
            intermediate_data_type,
            scale_data_type,
            output_data_type,
            accumulation_data_type,
            true,
            scale_free,
        )
        .map_err(QKVNormError::BackendError)?;
        Ok(Head {
            kernel,
            scales,
            config,
        })
    }

    pub fn encode(
        &self,
        qkv: &mut Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        if let Some(query) = &self.query {
            self.encode_head(query, qkv, batch_dim, 0, self.num_q_heads as u32, encoder);
        }
        if let Some(key) = &self.key {
            self.encode_head(key, qkv, batch_dim, self.num_q_heads as u32, self.num_kv_heads as u32, encoder);
        }
        if let Some(value) = &self.value {
            let value_offset = (self.num_q_heads + self.num_kv_heads) as u32;
            self.encode_head(value, qkv, batch_dim, value_offset, self.num_kv_heads as u32, encoder);
        }
        Ok(())
    }

    fn encode_head(
        &self,
        head: &Head<B>,
        qkv: &mut Allocation<B>,
        batch_dim: usize,
        range_start: u32,
        range_end: u32,
        encoder: &mut Encoder<B>,
    ) {
        head.kernel.encode(
            None::<&Allocation<B>>,
            head.scales.as_ref(),
            &mut *qkv,
            batch_dim as u32,
            self.num_q_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            head.config.epsilon,
            head.config.scale_offset.unwrap_or(0.0),
            range_start,
            range_end,
            head.config.upcast_mode == UpcastMode::FullLayer,
            encoder,
        );
    }
}
