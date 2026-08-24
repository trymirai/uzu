use crate::{
    backends::common::{
        Allocation, Backend, BufferArg, Encoder, Kernels,
        kernel::{attention_gemm::AttentionGemmCore, attention_gemm_grouped::AttentionGemmGroupedCore},
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

const SINGLE_PASS_KV_THRESHOLD: u32 = 1024;
const D256_SINGLE_PASS_KV_THRESHOLD: u32 = 150_000;
const D128_SHORT_KV_GEMM_THRESHOLD: u32 = 2_048;

pub struct AttentionCoreNewArguments {
    pub head_dim: u32,
    pub num_groups: u32,
    pub num_q_heads: u32,
    pub has_sinks: bool,
    pub is_kv_cache_ring: bool,
    pub is_causal: bool,
    pub is_trie: bool,
    pub sliding_window_size: Option<u32>,
    pub scale: Option<f32>,
    pub data_type: DataType,
}

pub struct AttentionCoreEncodeArguments<'a, B: Backend, KT: BufferArg<'a, B>, VT: BufferArg<'a, B>> {
    pub queries: &'a Allocation<B>,
    pub keys: KT,
    pub values: VT,
    pub suffix_length: u32,
    pub trie: Option<&'a Allocation<B>>,
    pub sinks: Option<&'a Allocation<B>>,
    pub state_type: &'a AttentionStateType,
}

pub struct AttentionCores<B: Backend> {
    head_dim: u32,
    grouped: Option<<B::Kernels as Kernels>::AttentionGemmGroupedCore>,
    gemm: Option<<B::Kernels as Kernels>::AttentionGemmCore>,
    fallback: Option<AttentionFallbackCore<B>>,
    two_pass: AttentionTwoPassCore<B>,
    single_pass: AttentionSinglePassCore<B>,
}

impl<B: Backend> AttentionCores<B> {
    pub fn new(
        arguments: AttentionCoreNewArguments,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        let grouped =
            if <<B::Kernels as Kernels>::AttentionGemmGroupedCore as AttentionGemmGroupedCore<B>>::is_supported(
                &arguments, context,
            )? {
                Some(<<B::Kernels as Kernels>::AttentionGemmGroupedCore as AttentionGemmGroupedCore<B>>::new(
                    context, &arguments,
                )?)
            } else {
                None
            };
        let gemm =
            if <<B::Kernels as Kernels>::AttentionGemmCore as AttentionGemmCore<B>>::is_supported(&arguments, context)?
            {
                Some(<<B::Kernels as Kernels>::AttentionGemmCore as AttentionGemmCore<B>>::new(context, &arguments)?)
            } else {
                None
            };
        let fallback = if AttentionFallbackCore::<B>::is_supported(&arguments) {
            Some(AttentionFallbackCore::new(&arguments, context)?)
        } else {
            None
        };
        let two_pass = AttentionTwoPassCore::new(&arguments, context)?;
        let single_pass = AttentionSinglePassCore::new(&arguments, context)?;

        Ok(Self {
            head_dim: arguments.head_dim,
            grouped,
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
        encoder.push_debug_group("attention core");

        let output = self.select_and_encode(arguments, encoder);

        encoder.pop_debug_group();

        output
    }

    fn select_and_encode<'a, KT: BufferArg<'a, B>, VT: BufferArg<'a, B>>(
        &self,
        arguments: AttentionCoreEncodeArguments<'a, B, KT, VT>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, <B as Backend>::Error> {
        let suffix_length = arguments.suffix_length;
        let kv_length = arguments.state_type.physical_prefix_length() + suffix_length;
        let is_kv_cache_ring = arguments.state_type.ring_params().is_some();

        if !is_kv_cache_ring {
            if let Some(grouped) = &self.grouped {
                if grouped.should_encode(suffix_length, kv_length) {
                    return grouped.encode(arguments, encoder);
                }

                // These measured sibling exceptions apply only when the
                // grouped core was built for this layer.
                if self.head_dim == 256
                    && (1..=16).contains(&suffix_length)
                    && kv_length > D256_SINGLE_PASS_KV_THRESHOLD
                {
                    return self.single_pass.encode(arguments, encoder);
                }
                if (self.head_dim == 256 && (9..=16).contains(&suffix_length))
                    || (self.head_dim == 128
                        && (9..=16).contains(&suffix_length)
                        && kv_length > D128_SHORT_KV_GEMM_THRESHOLD)
                {
                    return self.two_pass.encode(arguments, encoder);
                }
            }
        }

        if suffix_length > 8 {
            if let Some(gemm) = &self.gemm {
                return gemm.encode(arguments, encoder);
            }
            if let Some(fallback) = &self.fallback {
                return fallback.encode(arguments, encoder);
            }
        }

        if kv_length > SINGLE_PASS_KV_THRESHOLD {
            self.two_pass.encode(arguments, encoder)
        } else {
            self.single_pass.encode(arguments, encoder)
        }
    }
}

#[cfg(test)]
#[path = "../../../../../unit/encodable_block/attention_test.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../../../unit/encodable_block/attention_gemm_test.rs"]
mod gemm_tests;
