use parking_lot::Mutex;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, BufferArg, Encoder, Kernels,
        kernel::{
            AttentionFallbackScatterScoresKernel, AttentionFallbackScatterValuesKernel, SoftmaxKernel,
            matmul::{MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
        },
    },
    data_type::DataType,
    encodable_block::mixer::attention::core::{AttentionCoreEncodeArguments, AttentionCoreNewArguments},
};

pub struct AttentionFallbackCore<B: Backend> {
    head_dim: usize,
    num_groups: usize,
    num_q_heads: usize,
    sliding_window_size: Option<usize>,
    scale: Option<f32>,
    data_type: DataType,
    scatter_scores: <B::Kernels as Kernels>::AttentionFallbackScatterScoresKernel,
    scatter_values: <B::Kernels as Kernels>::AttentionFallbackScatterValuesKernel,
    softmax: <B::Kernels as Kernels>::SoftmaxKernel,
    matmul: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
}

impl<B: Backend> AttentionFallbackCore<B> {
    pub fn new(
        arguments: &AttentionCoreNewArguments,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        assert!(!arguments.is_trie, "trie not supported by attention fallback"); // Is it?

        let scatter_scores = <B::Kernels as Kernels>::AttentionFallbackScatterScoresKernel::new(
            context,
            arguments.data_type,
            arguments.is_kv_cache_ring,
            arguments.is_causal,
            arguments.is_trie,
            arguments.sliding_window_size.is_some(),
        )?;
        let scatter_values =
            <B::Kernels as Kernels>::AttentionFallbackScatterValuesKernel::new(context, arguments.data_type)?;
        let softmax = <B::Kernels as Kernels>::SoftmaxKernel::new(context, arguments.data_type, arguments.has_sinks)?;
        let matmul = Mutex::new(<B::Kernels as Kernels>::MatmulKernel::new(
            context,
            arguments.data_type,
            arguments.data_type,
            arguments.data_type,
        )?);

        Ok(Self {
            head_dim: arguments.head_dim,
            num_groups: arguments.num_groups,
            num_q_heads: arguments.num_q_heads,
            sliding_window_size: arguments.sliding_window_size,
            scale: arguments.scale,
            data_type: arguments.data_type,
            scatter_scores,
            scatter_values,
            softmax,
            matmul,
        })
    }

    pub fn encode<'a, KT: BufferArg<'a, B>, VT: BufferArg<'a, B>>(
        &self,
        arguments: AttentionCoreEncodeArguments<'a, B, KT, VT>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let sequence_length = arguments.state_type.physical_prefix_length() + arguments.suffix_length;
        let gqa_factor = self.num_q_heads / self.num_groups;
        let scale = self.scale.unwrap_or(1.0f32 / (self.head_dim as f32).sqrt());
        let dt_bytes = self.data_type.size_in_bytes();

        let mut output = encoder.allocate_constant(size_for_shape(
            &[arguments.suffix_length, self.num_q_heads, self.head_dim],
            self.data_type,
        ))?;
        let mut scores = encoder.allocate_scratch(size_for_shape(
            &[self.num_q_heads * arguments.suffix_length * sequence_length],
            self.data_type,
        ))?;
        let mut group_scores = encoder.allocate_scratch(size_for_shape(
            &[gqa_factor * arguments.suffix_length, sequence_length],
            self.data_type,
        ))?;

        for group_index in 0..self.num_groups {
            self.matmul.lock().encode(
                MatmulArguments {
                    a: arguments.queries,
                    a_offset: group_index * gqa_factor * arguments.suffix_length * self.head_dim * dt_bytes,
                    b: MatmulB::FullPrecision {
                        b: (arguments.keys, group_index * self.head_dim * dt_bytes),
                    },
                    b_leading_dimension: Some((self.num_groups * self.head_dim) as u32),
                    b_transpose: true,
                    d: &mut group_scores,
                    d_transform: MatmulDOps {
                        ab_scale: scale,
                        accumulate: false,
                        bias: None,
                        rht_factors: None,
                    },
                    m: (gqa_factor * arguments.suffix_length) as u32,
                    n: sequence_length as u32,
                    k: self.head_dim as u32,
                },
                encoder,
            )?;
            self.scatter_scores.encode(
                &group_scores,
                &mut scores,
                arguments.state_type.ring_params(),
                None::<&Allocation<B>>,
                self.sliding_window_size.map(|sliding_window_size| sliding_window_size as u32),
                group_index as u32,
                gqa_factor as u32,
                sequence_length as u32,
                arguments.suffix_length as u32,
                (gqa_factor * arguments.suffix_length * sequence_length) as u32,
                encoder,
            );
        }

        self.softmax.encode(
            &mut scores,
            arguments.sinks,
            sequence_length as u32,
            self.num_q_heads as u32,
            arguments.suffix_length as u32,
            encoder,
        );

        let mut group_output = encoder
            .allocate_scratch(size_for_shape(&[gqa_factor * arguments.suffix_length, self.head_dim], self.data_type))?;

        for group_index in 0..self.num_groups {
            self.matmul.lock().encode(
                MatmulArguments {
                    a: &scores,
                    a_offset: group_index * gqa_factor * arguments.suffix_length * sequence_length * dt_bytes,
                    b: MatmulB::FullPrecision {
                        b: (arguments.values, group_index * self.head_dim * dt_bytes),
                    },
                    b_leading_dimension: Some((self.num_groups * self.head_dim) as u32),
                    b_transpose: false,
                    d: &mut group_output,
                    d_transform: MatmulDOps::none(),
                    m: (gqa_factor * arguments.suffix_length) as u32,
                    n: self.head_dim as u32,
                    k: sequence_length as u32,
                },
                encoder,
            )?;
            self.scatter_values.encode(
                &group_output,
                &mut output,
                group_index as u32,
                gqa_factor as u32,
                arguments.suffix_length as u32,
                self.num_q_heads as u32,
                self.head_dim as u32,
                (gqa_factor * arguments.suffix_length * self.head_dim) as u32,
                encoder,
            );
        }

        Ok(output)
    }
}
