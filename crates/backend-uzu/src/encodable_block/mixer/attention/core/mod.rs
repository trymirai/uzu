use crate::{
    backends::common::{Allocation, Backend, BufferArg, Encoder},
    data_type::DataType,
    encodable_block::mixer::attention::{
        core::{
            fallback::AttentionFallbackCore, gemm::AttentionGemmCore, single_pass::AttentionSinglePassCore,
            two_pass::AttentionTwoPassCore,
        },
        state::AttentionStateType,
    },
};

mod fallback;
mod gemm;
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

pub struct AttentionCores<B: Backend> {
    gemm: Option<AttentionGemmCore<B>>,
    fallback: Option<AttentionFallbackCore<B>>,
    two_pass: AttentionTwoPassCore<B>,
    single_pass: AttentionSinglePassCore<B>,
}

impl<B: Backend> AttentionCores<B> {
    pub fn new(
        arguments: AttentionCoreNewArguments,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        let gemm = if matches!(arguments.head_dim, 64 | 128 | 256) {
            Some(AttentionGemmCore::new(&arguments, context)?)
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
            gemm.encode(arguments, encoder)
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
}

#[cfg(test)]
#[path = "../../../../../unit/encodable_block/attention_test.rs"]
mod tests;
