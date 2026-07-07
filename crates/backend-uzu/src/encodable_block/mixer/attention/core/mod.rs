use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::AttnParams,
        kernel::{BufferArg, attention_gemm::AttentionGemmDispatch},
    },
    data_type::DataType,
    encodable_block::mixer::attention::{
        core::{fallback::AttentionFallbackCore, single_pass::AttentionSinglePassCore, two_pass::AttentionTwoPassCore},
        state::AttentionStateType,
    },
};

mod fallback;
mod single_pass;
mod two_pass;

pub struct AttentionCoreNewArguments {
    pub head_dim: usize,
    pub num_groups: usize,
    pub num_q_heads: usize,
    pub has_sinks: bool,
    pub is_kv_cache_ring: bool,
    pub is_causal: bool,
    pub is_trie: bool,
    pub sliding_window_size: Option<usize>,
    pub scale: Option<f32>,
    pub data_type: DataType,
}

pub struct AttentionCoreEncodeArguments<'a, B: Backend, KT: BufferArg<'a, B>, VT: BufferArg<'a, B>> {
    pub queries: &'a Allocation<B>,
    pub keys: KT,
    pub values: VT,
    pub suffix_length: usize,
    pub trie: Option<&'a Allocation<B>>,
    pub sinks: Option<&'a Allocation<B>>,
    pub state_type: &'a AttentionStateType,
}

pub struct AttentionCore<B: Backend> {
    gemm: Option<<B::Kernels as Kernels>::AttentionGemmDispatch>,
    fallback: Option<AttentionFallbackCore<B>>,
    two_pass: AttentionTwoPassCore<B>,
    single_pass: AttentionSinglePassCore<B>,
    head_dim: usize,
    num_groups: usize,
    num_q_heads: usize,
    gemm_key_tile: usize,
    gemm_query_tile: usize,
    sliding_window_size: Option<usize>,
    scale: Option<f32>,
    data_type: DataType,
}

impl<B: Backend> AttentionCore<B> {
    pub fn new(
        arguments: AttentionCoreNewArguments,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        let gemm = if matches!(arguments.head_dim, 64 | 128 | 256) {
            let key_tile = if arguments.head_dim < 128 {
                32
            } else {
                16
            };
            Some(<B::Kernels as Kernels>::AttentionGemmDispatch::new(
                context,
                arguments.data_type,
                key_tile as u32,
                arguments.head_dim as u32,
                arguments.is_kv_cache_ring,
                arguments.is_causal,
                arguments.is_trie,
                arguments.sliding_window_size.is_some(),
                arguments.has_sinks,
            )?)
        } else {
            None
        };
        let fallback = if arguments.head_dim == 512 && !arguments.is_trie {
            Some(AttentionFallbackCore::new(&arguments, context)?)
        } else {
            None
        };
        let two_pass = AttentionTwoPassCore::new(&arguments, context)?;
        let single_pass = AttentionSinglePassCore::new(&arguments, context)?;

        Ok(Self {
            gemm,
            fallback,
            two_pass,
            single_pass,
            head_dim: arguments.head_dim,
            num_groups: arguments.num_groups,
            num_q_heads: arguments.num_q_heads,
            gemm_key_tile: if arguments.head_dim < 128 {
                32
            } else {
                16
            },
            gemm_query_tile: 32,
            sliding_window_size: arguments.sliding_window_size,
            scale: arguments.scale,
            data_type: arguments.data_type,
        })
    }

    pub fn encode<'a, KT: BufferArg<'a, B>, VT: BufferArg<'a, B>>(
        &self,
        arguments: AttentionCoreEncodeArguments<'a, B, KT, VT>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, <B as Backend>::Error> {
        if arguments.suffix_length > 8
            && let Some(gemm) = &self.gemm
        {
            self.encode_gemm(gemm, arguments, encoder)
        } else if arguments.suffix_length > 8
            && let Some(fallback) = &self.fallback
        {
            fallback.encode(arguments, encoder)
        } else if arguments.state_type.physical_prefix_length() + arguments.suffix_length > 1024 {
            self.two_pass.encode(arguments, encoder)
        } else {
            self.single_pass.encode(arguments, encoder)
        }
    }

    fn encode_gemm<'a, KT: BufferArg<'a, B>, VT: BufferArg<'a, B>>(
        &self,
        gemm: &<B::Kernels as Kernels>::AttentionGemmDispatch,
        arguments: AttentionCoreEncodeArguments<'a, B, KT, VT>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, <B as Backend>::Error> {
        let mut output = encoder.allocate_constant(size_for_shape(
            &[arguments.suffix_length, self.num_q_heads, self.head_dim],
            self.data_type,
        ))?;
        let key_len = arguments.state_type.physical_prefix_length() + arguments.suffix_length;
        gemm.encode(
            arguments.queries,
            arguments.keys,
            arguments.values,
            &mut output,
            AttnParams {
                q_strides: [0, (arguments.suffix_length * self.head_dim) as u64, self.head_dim as u64],
                k_strides: [0, self.head_dim as u64, (self.num_groups * self.head_dim) as u64],
                v_strides: [0, self.head_dim as u64, (self.num_groups * self.head_dim) as u64],
                o_strides: [0, self.head_dim as u64, (self.num_q_heads * self.head_dim) as u64],
                gqa_factor: (self.num_q_heads / self.num_groups) as u32,
                scale: self.scale.unwrap_or(1.0f32 / (self.head_dim as f32).sqrt()),
                q_len: arguments.suffix_length as u32,
                k_len: key_len as u32,
                q_off: arguments.state_type.physical_prefix_length() as u32,
                nq_aligned: (arguments.suffix_length / self.gemm_query_tile) as u32,
                q_rem: (arguments.suffix_length % self.gemm_query_tile) as u32,
                nk: key_len.div_ceil(self.gemm_key_tile) as u32,
                nk_aligned: (key_len / self.gemm_key_tile) as u32,
                k_rem: (key_len % self.gemm_key_tile) as u32,
            },
            arguments.state_type.ring_params(),
            arguments.trie,
            self.sliding_window_size.map(|sliding_window_size| sliding_window_size as u32),
            arguments.sinks,
            self.num_q_heads as u32,
            arguments.suffix_length as u32,
            encoder,
        );

        Ok(output)
    }
}

#[cfg(test)]
#[path = "../../../../../unit/encodable_block/attention_gemm_test.rs"]
mod gemm_tests;

#[cfg(test)]
#[path = "../../../../../unit/encodable_block/attention_test.rs"]
mod tests;
