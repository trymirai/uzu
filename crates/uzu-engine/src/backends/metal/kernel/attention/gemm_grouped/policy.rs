use crate::{
    backends::metal::context::MetalContext, data_type::DataType,
    encodable_block::mixer::attention::core::AttentionCoreNewArguments,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MaskKind {
    #[default]
    None,
    Causal,
    Trie,
}

impl MaskKind {
    pub fn for_attention(
        is_causal: bool,
        is_trie: bool,
    ) -> Option<Self> {
        match (is_causal, is_trie) {
            (true, true) => Some(Self::Trie),
            (true, false) => Some(Self::Causal),
            (false, false) => Some(Self::None),
            (false, true) => None,
        }
    }

    pub fn is_causal(self) -> bool {
        !matches!(self, Self::None)
    }

    pub fn is_trie(self) -> bool {
        matches!(self, Self::Trie)
    }
}

const GEMM_GROUPED_HEAD_DIMS: [u32; 2] = [128, 256];
const GEMM_GROUPED_DECODE_SUFFIX_MIN: u32 = 16;
const GEMM_GROUPED_DECODE_SUFFIX_MAX: u32 = 64;
const GEMM_GROUPED_PREFILL_SUFFIX_MAX: u32 = 1024;
const GEMM_GROUPED_MIN_KV_LENGTH: u32 = 1024;
pub const MAX_TRIE_SUFFIX: u32 = 64;

pub fn is_supported(
    arguments: &AttentionCoreNewArguments,
    context: &MetalContext,
) -> bool {
    GEMM_GROUPED_HEAD_DIMS.contains(&arguments.head_dim)
        && arguments.data_type == DataType::BF16
        && context.supports_mxu()
        && !arguments.has_sinks
        && !arguments.is_kv_cache_ring
        && arguments.sliding_window_size.is_none()
        && arguments.scale.is_none_or(|scale| scale > 0.0)
        && arguments.num_groups > 0
        && arguments.num_q_heads.is_multiple_of(arguments.num_groups)
        && MaskKind::for_attention(arguments.is_causal, arguments.is_trie).is_some()
}

pub fn should_encode(
    head_dim: u32,
    mask: MaskKind,
    suffix_length: u32,
    kv_length: u32,
) -> bool {
    if kv_length >= GEMM_GROUPED_MIN_KV_LENGTH
        && ((GEMM_GROUPED_DECODE_SUFFIX_MIN..=GEMM_GROUPED_DECODE_SUFFIX_MAX).contains(&suffix_length)
            || (head_dim == 256
                && mask == MaskKind::Causal
                && (GEMM_GROUPED_DECODE_SUFFIX_MAX + 1..=GEMM_GROUPED_PREFILL_SUFFIX_MAX).contains(&suffix_length)))
    {
        return true;
    }
    false
}

type MeasuredSplits = (u32, u32, &'static [(u32, u32)]);

const MEASURED_SUFFIX_MIN: u32 = 16;
const MEASURED_SUFFIX_MAX: u32 = 64;

// TODO: validate this table per chip (M1-M5) before changing the policy.
// TODO: add a simdgroup/non-MXU implementation before widening availability.
const MEASURED_TG_PER_CORE_TENTHS: &[MeasuredSplits] = &[
    (256, 16, &[(0, 24), (5120, 18), (131072, 39)]),
    (256, 32, &[(0, 24), (5120, 18), (32768, 60)]),
    (256, 64, &[(0, 24), (5120, 60)]),
    (128, 16, &[(0, 16), (51200, 40)]),
    (128, 32, &[(0, 16), (5120, 40), (51200, 80)]),
    (128, 64, &[(0, 32), (5120, 40), (32768, 80)]),
];
const TG_PER_CORE_FALLBACK: u32 = 6;
const TENTHS_PER_RATIO: u32 = 10;

fn tg_per_core_tenths_for(
    steps: &[(u32, u32)],
    kv_length: u32,
) -> u32 {
    steps.iter().rev().find(|(minimum_kv, _)| kv_length >= *minimum_kv).map_or(steps[0].1, |(_, tenths)| *tenths)
}

fn splits_for_ratio(
    tg_per_core_tenths: u32,
    gpu_core_count: u32,
    unsplit_threadgroups: u32,
) -> u32 {
    (tg_per_core_tenths * gpu_core_count.max(1)).div_ceil(TENTHS_PER_RATIO * unsplit_threadgroups.max(1))
}

fn tabled_splits(
    head_dim: u32,
    suffix_length: u32,
    kv_length: u32,
    unsplit_threadgroups: u32,
    gpu_core_count: u32,
) -> Option<u32> {
    if !(MEASURED_SUFFIX_MIN..=MEASURED_SUFFIX_MAX).contains(&suffix_length) {
        return None;
    }
    MEASURED_TG_PER_CORE_TENTHS
        .iter()
        .filter(|(row_head_dim, _, _)| *row_head_dim == head_dim)
        .min_by_key(|(_, row_suffix, _)| (row_suffix.abs_diff(suffix_length), u32::MAX - row_suffix))
        .map(|(_, _, steps)| {
            splits_for_ratio(tg_per_core_tenths_for(steps, kv_length), gpu_core_count, unsplit_threadgroups)
        })
}

pub fn choose_splits(
    head_dim: u32,
    suffix_length: u32,
    kv_length: u32,
    unsplit_threadgroups: u32,
    block_k: u32,
    gpu_core_count: u32,
) -> u32 {
    let splits = tabled_splits(head_dim, suffix_length, kv_length, unsplit_threadgroups, gpu_core_count)
        .unwrap_or_else(|| {
            splits_for_ratio(TENTHS_PER_RATIO * TG_PER_CORE_FALLBACK, gpu_core_count, unsplit_threadgroups)
        });
    splits.clamp(1, kv_length.div_ceil(block_k).max(1))
}

#[cfg(test)]
#[path = "../../../../../../unit/backends/metal/kernel/attention_gemm_grouped_policy_test.rs"]
mod tests;
