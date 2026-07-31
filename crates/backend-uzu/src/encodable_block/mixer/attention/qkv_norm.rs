use thiserror::Error;

use crate::{
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{Kernels, QKVNormKernel},
    },
    config::normalization::{NormalizationConfig, UpcastMode},
    data_type::DataType,
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
        let scales = if let Some(scales_leaf) = scales_leaf {
            Some(parameter_tree.leaf(scales_leaf)?.validate(&[head_dim], DataType::F32)?.read_allocation()?)
        } else {
            None
        };
        let kernel = <B::Kernels as Kernels>::QKVNormKernel::new(
            context,
            intermediate_data_type,
            DataType::F32,
            intermediate_data_type,
            DataType::F32,
            true,
            scales.is_none(),
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
        self.encode_packed(qkv, batch_dim, self.num_q_heads, encoder)
    }

    pub fn encode_key_value(
        &self,
        key_value: &mut Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        self.encode_packed(key_value, batch_dim, 0, encoder)
    }

    fn encode_packed(
        &self,
        buffer: &mut Allocation<B>,
        batch_dim: usize,
        q_heads: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let kv = self.num_kv_heads;
        let total_heads = q_heads + 2 * kv;
        let heads = [(&self.query, 0, q_heads), (&self.key, q_heads, kv), (&self.value, q_heads + kv, kv)];
        for (head, head_offset, head_count) in heads {
            let Some(head) = head else {
                continue;
            };
            if head_count == 0 {
                continue;
            }
            head.kernel.encode(
                None::<&Allocation<B>>,
                head.scales.as_ref(),
                &mut *buffer,
                batch_dim as u32,
                total_heads as u32,
                self.head_dim as u32,
                head.config.epsilon,
                head.config.scale_offset.unwrap_or(0.0),
                head_offset as u32,
                head_count as u32,
                head.config.upcast_mode == UpcastMode::FullLayer,
                encoder,
            );
        }
        Ok(())
    }
}
