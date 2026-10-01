use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferMut, CommandBuffer, CommandBufferEncoding,
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
    scales: Option<B::GlobalBuffer>,
    config: NormalizationConfig,
}

pub struct QKVNorm<B: Backend> {
    query: Option<Head<B>>,
    key: Option<Head<B>>,
    value: Option<Head<B>>,
    num_q_heads: u32,
    num_kv_heads: u32,
    projection_row_stride: u32,
    head_dim: u32,
}

impl<B: Backend> QKVNorm<B> {
    pub fn new(
        context: &B::Context,
        intermediate_data_type: DataType,
        query_config: Option<NormalizationConfig>,
        key_config: Option<NormalizationConfig>,
        value_config: Option<NormalizationConfig>,
        parameter_tree: &ParameterTree<B>,
        num_q_heads: u32,
        num_kv_heads: u32,
        projection_row_stride: u32,
        head_dim: u32,
    ) -> Result<Self, QKVNormError<B>> {
        let query = query_config
            .map(|cfg| {
                Self::build_head(
                    context,
                    intermediate_data_type,
                    cfg,
                    Some(&parameter_tree.subtree("query_norm")),
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
                    Some(&parameter_tree.subtree("key_norm")),
                    head_dim,
                )
            })
            .transpose()?;
        let value = value_config
            .map(|cfg| Self::build_head(context, intermediate_data_type, cfg, None, head_dim))
            .transpose()?;

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

    fn build_head(
        context: &B::Context,
        intermediate_data_type: DataType,
        config: NormalizationConfig,
        parameter_tree: Option<&ParameterTree<B>>,
        head_dim: u32,
    ) -> Result<Head<B>, QKVNormError<B>> {
        let scales = if config.has_scale {
            Some(
                parameter_tree
                    .expect("scaled norm requires parameter tree")
                    .leaf("scales")?
                    .validate(&[head_dim], DataType::F32)?
                    .read_buffer()?,
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
        Ok(Head {
            kernel,
            scales,
            config,
        })
    }

    pub fn encode(
        &self,
        qkvg: impl BufferMut<Backend = B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        self.encode_packed(qkvg, batch_dim, self.num_q_heads, self.projection_row_stride, command_buffer)
    }

    pub fn encode_key_value(
        &self,
        key_value: impl BufferMut<Backend = B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        self.encode_packed(key_value, batch_dim, 0, 2 * self.num_kv_heads * self.head_dim, command_buffer)
    }

    fn encode_packed(
        &self,
        mut buffer: impl BufferMut<Backend = B>,
        batch_dim: u32,
        q_heads: u32,
        input_row_stride: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let packed_row_width = (q_heads + 2 * self.num_kv_heads) * self.head_dim;
        assert!(
            input_row_stride >= packed_row_width,
            "QKV norm input row stride ({input_row_stride}) is smaller than its packed row width ({packed_row_width})"
        );

        command_buffer.push_debug_group("qkv norm");

        let kv = self.num_kv_heads;
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

        command_buffer.pop_debug_group();

        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../../unit/encodable_block/attention/qkv_norm_test.rs"]
mod tests;
