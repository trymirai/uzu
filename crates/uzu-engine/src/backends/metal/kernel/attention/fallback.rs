use parking_lot::Mutex;

use crate::{
    backends::{
        common::{
            Allocation, BufferArg, Encoder, Kernels,
            kernel::{
                AttentionArguments, AttentionKernelConfig, SoftmaxKernel,
                matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            },
        },
        metal::{
            Metal,
            context::MetalContext,
            error::MetalError,
            kernel::{
                AttentionFallbackScatterScoresMetalKernel, AttentionFallbackScatterValuesMetalKernel, MetalKernels,
            },
        },
    },
    data_type::DataType,
};

const HEAD_DIM: u32 = 512;

pub struct AttentionFallback {
    head_dim: u32,
    num_groups: u32,
    num_q_heads: u32,
    sliding_window_size: Option<u32>,
    scale: Option<f32>,
    data_type: DataType,
    scatter_scores: AttentionFallbackScatterScoresMetalKernel,
    scatter_values: AttentionFallbackScatterValuesMetalKernel,
    softmax: <MetalKernels as Kernels>::SoftmaxKernel,
    matmul: Mutex<<MetalKernels as Kernels>::MatmulKernel>,
}

impl AttentionFallback {
    pub fn is_supported(config: &AttentionKernelConfig) -> bool {
        config.head_dim == HEAD_DIM && matches!(config.data_type, DataType::BF16 | DataType::F32)
    }

    pub fn new(
        config: &AttentionKernelConfig,
        context: &MetalContext,
    ) -> Result<Self, MetalError> {
        let scatter_scores = AttentionFallbackScatterScoresMetalKernel::new(
            context,
            config.data_type,
            config.is_kv_cache_ring,
            config.is_causal,
            false,
            config.sliding_window_size.is_some(),
        )?;
        let scatter_values = AttentionFallbackScatterValuesMetalKernel::new(context, config.data_type)?;
        let softmax = <<MetalKernels as Kernels>::SoftmaxKernel as SoftmaxKernel>::new(
            context,
            config.data_type,
            config.has_sinks,
        )?;
        let matmul = Mutex::new(<<MetalKernels as Kernels>::MatmulKernel as MatmulKernel>::new(
            context,
            config.data_type,
            config.data_type,
            config.data_type,
        )?);
        Ok(Self {
            head_dim: config.head_dim,
            num_groups: config.num_groups,
            num_q_heads: config.num_q_heads,
            sliding_window_size: config.sliding_window_size,
            scale: config.scale,
            data_type: config.data_type,
            scatter_scores,
            scatter_values,
            softmax,
            matmul,
        })
    }

    pub fn encode<'a, KT: BufferArg<'a, Metal>, VT: BufferArg<'a, Metal>>(
        &self,
        arguments: AttentionArguments<'a, Metal, KT, VT>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<Allocation<Metal>, MetalError> {
        assert!(arguments.trie.is_none(), "fallback does not support trie");
        let suffix_length = arguments.suffix_length;
        let sequence_length = arguments.cache.prefix_len() + suffix_length;
        let gqa_factor = self.num_q_heads / self.num_groups;
        let scale = self.scale.unwrap_or(1.0 / (self.head_dim as f32).sqrt());
        let dt_bytes = self.data_type.size_in_bytes();
        let head_dim_bytes = self.head_dim as usize * dt_bytes;
        let group_rows = (gqa_factor * suffix_length) as usize;
        let mut output =
            encoder.allocate_constant_for_shape(&[suffix_length, self.num_q_heads, self.head_dim], self.data_type)?;
        let mut scores =
            encoder.allocate_scratch_for_shape(&[self.num_q_heads, suffix_length, sequence_length], self.data_type)?;
        let mut group_scores =
            encoder.allocate_scratch_for_shape(&[gqa_factor * suffix_length, sequence_length], self.data_type)?;

        for group_index in 0..self.num_groups {
            self.matmul.lock().encode(
                MatmulArguments {
                    a: MatmulA::FullPrecision {
                        values: arguments.queries,
                        offset: group_index as usize * group_rows * head_dim_bytes,
                    },
                    b: MatmulB::FullPrecision {
                        b: (arguments.keys, group_index as usize * head_dim_bytes),
                    },
                    b_leading_dimension: Some(self.num_groups * self.head_dim),
                    b_transpose: true,
                    d: &mut group_scores,
                    d_transform: MatmulDOps {
                        ab_scale: scale,
                        ..MatmulDOps::none()
                    },
                    gather_indices: None,
                    m: gqa_factor * suffix_length,
                    n: sequence_length,
                    k: self.head_dim,
                },
                encoder,
            )?;
            self.scatter_scores.encode(
                &group_scores,
                &mut scores,
                arguments.cache.ring_params(),
                None::<&Allocation<Metal>>,
                self.sliding_window_size,
                group_index,
                gqa_factor,
                sequence_length,
                suffix_length,
                gqa_factor * suffix_length * sequence_length,
                encoder,
            );
        }

        self.softmax.encode(&mut scores, arguments.sinks, sequence_length, self.num_q_heads, suffix_length, encoder);
        let mut group_output =
            encoder.allocate_scratch_for_shape(&[gqa_factor * suffix_length, self.head_dim], self.data_type)?;
        for group_index in 0..self.num_groups {
            self.matmul.lock().encode(
                MatmulArguments {
                    a: MatmulA::FullPrecision {
                        values: &scores,
                        offset: group_index as usize * group_rows * sequence_length as usize * dt_bytes,
                    },
                    b: MatmulB::FullPrecision {
                        b: (arguments.values, group_index as usize * head_dim_bytes),
                    },
                    b_leading_dimension: Some(self.num_groups * self.head_dim),
                    b_transpose: false,
                    d: &mut group_output,
                    d_transform: MatmulDOps::none(),
                    gather_indices: None,
                    m: gqa_factor * suffix_length,
                    n: self.head_dim,
                    k: sequence_length,
                },
                encoder,
            )?;
            self.scatter_values.encode(
                &group_output,
                &mut output,
                group_index,
                gqa_factor,
                suffix_length,
                self.num_q_heads,
                self.head_dim,
                gqa_factor * suffix_length * self.head_dim,
                encoder,
            );
        }
        Ok(output)
    }
}
