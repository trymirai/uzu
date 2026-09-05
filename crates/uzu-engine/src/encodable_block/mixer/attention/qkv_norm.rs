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
                    .read_allocation()?,
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
        qkvg: &mut Allocation<B>,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        self.encode_packed(qkvg, batch_dim, self.num_q_heads, self.projection_row_stride, encoder)
    }

    pub fn encode_key_value(
        &self,
        key_value: &mut Allocation<B>,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        self.encode_packed(key_value, batch_dim, 0, 2 * self.num_kv_heads * self.head_dim, encoder)
    }

    fn encode_packed(
        &self,
        buffer: &mut Allocation<B>,
        batch_dim: u32,
        q_heads: u32,
        input_row_stride: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let packed_row_width = (q_heads + 2 * self.num_kv_heads) * self.head_dim;
        assert!(
            input_row_stride >= packed_row_width,
            "QKV norm input row stride ({input_row_stride}) is smaller than its packed row width ({packed_row_width})"
        );

        encoder.push_debug_group("qkv norm");

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
                None::<&Allocation<B>>,
                head.scales.as_ref(),
                &mut *buffer,
                batch_dim,
                input_row_stride,
                self.head_dim,
                head.config.epsilon,
                head.config.scale_offset.unwrap_or(0.0),
                head_offset,
                head_count,
                head.config.upcast_mode == UpcastMode::FullLayer,
                encoder,
            );
        }

        encoder.pop_debug_group();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use uzu_engine_macros::uzu_test;

    use super::QKVNorm;
    use crate::{
        backends::common::{Backend, Encoder},
        config::normalization::{NormalizationConfig, UpcastMode},
        data_type::DataType,
        tests::{
            assert::assert_eq_float,
            helpers::{alloc_allocation_with_data, allocation_to_vec, create_context, for_each_backend},
        },
    };

    fn run_key_value_row_stride_test<B: Backend>() {
        const BATCH_SIZE: u32 = 2;
        const NUM_KV_HEADS: u32 = 2;
        const HEAD_DIM: u32 = 4;
        const ROW_WIDTH: u32 = 2 * NUM_KV_HEADS * HEAD_DIM;
        const EPSILON: f32 = 1e-6;

        let context = create_context::<B>();
        let config = NormalizationConfig {
            epsilon: EPSILON,
            scale_offset: None,
            upcast_mode: UpcastMode::OnlyNormalization,
            subtract_mean: false,
            has_scale: false,
            has_biases: false,
        };
        let key = QKVNorm::<B>::build_head(&context, DataType::F32, config, None, HEAD_DIM)
            .expect("failed to construct key norm");
        let norm = QKVNorm {
            query: None,
            key: Some(key),
            value: None,
            num_q_heads: 0,
            num_kv_heads: NUM_KV_HEADS,
            projection_row_stride: ROW_WIDTH,
            head_dim: HEAD_DIM,
        };

        let row_width = ROW_WIDTH as usize;
        let head_dim = HEAD_DIM as usize;
        let input = (0..BATCH_SIZE as usize * row_width).map(|index| 1.0 + index as f32 * 0.125).collect::<Vec<_>>();
        let mut expected = input.clone();
        for batch in 0..BATCH_SIZE as usize {
            for head in 0..NUM_KV_HEADS as usize {
                let start = batch * row_width + head * head_dim;
                let mean_square =
                    input[start..start + head_dim].iter().map(|value| value * value).sum::<f32>() / HEAD_DIM as f32;
                let inverse_rms = (mean_square + EPSILON).sqrt().recip();
                for index in start..start + head_dim {
                    expected[index] *= inverse_rms;
                }
            }
        }

        let mut key_value = alloc_allocation_with_data::<B, f32>(&context, &input);
        let mut encoder = Encoder::new(context.as_ref()).expect("failed to create encoder");
        norm.encode_key_value(&mut key_value, BATCH_SIZE, &mut encoder).expect("failed to encode key/value norm");
        encoder.end_encoding().submit().wait_until_completed().expect("failed to execute key/value norm");

        let output = allocation_to_vec::<B, f32>(&key_value);
        assert_eq_float(&expected, &output, 1e-5, "key/value norm row stride mismatch");
    }

    #[uzu_test]
    fn test_key_value_norm_uses_element_row_stride() {
        for_each_backend!(|B| run_key_value_row_stride_test::<B>());
    }
}
