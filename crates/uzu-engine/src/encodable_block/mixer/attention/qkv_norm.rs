use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferMut, CommandBuffer,
        kernel::{Kernels, QKVNormKernel},
    },
    config::normalization::{NormalizationConfig, UpcastMode},
    data_type::DataType,
    safetensors::{ParameterLoaderError, ParameterTree},
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
    scales: Option<B::GlobalGpuBuffer>,
    config: NormalizationConfig,
}

pub struct QKVNorm<B: Backend> {
    query: Option<Head<B>>,
    key: Option<Head<B>>,
    value: Option<Head<B>>,
    num_q_heads: u32,
    num_kv_heads: Option<u32>,
    projection_row_stride: u32,
    head_dim: u32,
}

impl<B: Backend> QKVNorm<B> {
    pub async fn new(
        num_q_heads: u32,
        num_kv_heads: Option<u32>,
        projection_row_stride: u32,
        head_dim: u32,
        intermediate_data_type: DataType,
        query_config: Option<NormalizationConfig>,
        key_config: Option<NormalizationConfig>,
        value_config: Option<NormalizationConfig>,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, QKVNormError<B>> {
        let build_head = async |config: Option<NormalizationConfig>, parameter_name: Option<&str>| {
            let Some(config) = config else {
                return Ok::<_, QKVNormError<B>>(None);
            };
            let scales = if config.has_scale {
                Some(
                    parameter_tree
                        .subtree(parameter_name.expect("scaled norm requires parameter tree"))
                        .read_buffer("scales", &[head_dim], DataType::F32)
                        .await?,
                )
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
                scales.is_some(),
            )
            .map_err(QKVNormError::BackendError)?;
            Ok(Some(Head {
                kernel,
                scales,
                config,
            }))
        };

        let query = build_head(query_config, Some("query_norm")).await?;
        let key = build_head(key_config, Some("key_norm")).await?;
        let value = build_head(value_config, None).await?;
        Ok(Self {
            query,
            key,
            value,
            num_q_heads,
            num_kv_heads,
            projection_row_stride,
            head_dim,
        })
    }

    pub fn encode(
        &self,
        qkvg: impl BufferMut<Backend = B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        self.encode_packed(qkvg, batch_dim, self.num_q_heads, self.projection_row_stride, command_buffer)
    }

    pub fn encode_key_value(
        &self,
        key_value: impl BufferMut<Backend = B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        let num_kv_heads = self.num_kv_heads.expect("KV norm requires KV heads");
        self.encode_packed(key_value, batch_dim, 0, 2 * num_kv_heads * self.head_dim, command_buffer)
    }

    fn encode_packed(
        &self,
        mut buffer: impl BufferMut<Backend = B>,
        batch_dim: u32,
        q_heads: u32,
        input_row_stride: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        let kv = self.num_kv_heads.unwrap_or(0);
        let packed_row_width = (q_heads + 2 * kv) * self.head_dim;
        assert!(
            input_row_stride >= packed_row_width,
            "QKV norm input row stride ({input_row_stride}) is smaller than its packed row width ({packed_row_width})"
        );

        let heads = [(&self.query, 0, q_heads), (&self.key, q_heads, kv), (&self.value, q_heads + kv, kv)];
        for (head, head_offset, head_count) in heads {
            let Some(head) = head else {
                continue;
            };
            if head_count == 0 {
                continue;
            }
            head.kernel.encode(
                None::<&B::ScratchBuffer>,
                head.scales.as_ref(),
                buffer.reborrow(),
                batch_dim,
                input_row_stride,
                self.head_dim,
                head.config.epsilon,
                head.config.scale_offset.unwrap_or(0.0),
                head_offset,
                head_count,
                head.config.upcast_mode == UpcastMode::FullLayer,
                command_buffer,
            );
        }
    }
}

#[cfg(test)]
#[path = "../../../../unit/encodable_block/attention/qkv_norm_test.rs"]
mod tests;
