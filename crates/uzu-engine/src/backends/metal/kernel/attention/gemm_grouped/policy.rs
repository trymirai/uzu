use super::MaskKind;
use crate::{
    backends::{common::kernel::AttentionKernelConfig, metal::context::MetalContext},
    data_type::DataType,
};

const GEMM_GROUPED_HEAD_DIMS: [u32; 2] = [128, 256];
const GEMM_GROUPED_DECODE_SUFFIX_MIN: u32 = 2;
const GEMM_GROUPED_DECODE_SUFFIX_MAX: u32 = 64;
const GEMM_GROUPED_PREFILL_SUFFIX_MAX: u32 = 1024;
const GEMM_GROUPED_MIN_KV_LENGTH: u32 = 1024;
pub const MAX_TRIE_SUFFIX: u32 = 64;

pub fn is_supported(
    arguments: &AttentionKernelConfig,
    context: &MetalContext,
) -> bool {
    GEMM_GROUPED_HEAD_DIMS.contains(&arguments.head_dim)
        && arguments.data_type == DataType::BF16
        && context.supports_mxu
        && !arguments.has_sinks
        && !arguments.is_kv_cache_ring
        && arguments.sliding_window_size.is_none()
        && arguments.scale.is_none_or(|scale| scale > 0.0)
        && arguments.num_groups > 0
        && arguments.num_q_heads > 0
        && arguments.num_q_heads.is_multiple_of(arguments.num_groups)
}

pub fn should_encode(
    head_dim: u32,
    mask: MaskKind,
    suffix_length: u32,
    kv_length: u32,
) -> bool {
    kv_length >= GEMM_GROUPED_MIN_KV_LENGTH
        && ((GEMM_GROUPED_DECODE_SUFFIX_MIN..=GEMM_GROUPED_DECODE_SUFFIX_MAX).contains(&suffix_length)
            || (head_dim == 256
                && mask == MaskKind::Causal
                && (GEMM_GROUPED_DECODE_SUFFIX_MAX + 1..=GEMM_GROUPED_PREFILL_SUFFIX_MAX).contains(&suffix_length)))
}

type MeasuredSplits = (u32, u32, &'static [(u32, u32)]);

// Reuse S=16 tuning values to choose KV splits for suffixes 2..15.
const MEASURED_SUFFIX_MIN: u32 = 2;
const MEASURED_SUFFIX_MAX: u32 = 64;

// TODO: validate this table per chip (M1-M5) before changing the policy.
// TODO: add a simdgroup/non-MXU implementation before widening availability.
// TODO: measure the D128/D256 crossover before widening suffix ranges.
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
const PREFILL_SPLIT_KEYS: u32 = 2048;
const PREFILL_SPLIT_MIN_KV: u32 = 16384;
const PREFILL_SPLIT_SCRATCH_BYTES: u64 = 256 << 20;
const PREFILL_MIN_SPLITS: u32 = 2;
const PARTIAL_STATS_PER_ROW: u64 = 2; // max and sum

pub struct SplitGeometry {
    pub head_dim: u32,
    pub num_q_heads: u32,
    pub num_groups: u32,
    pub block_rows: u32,
    pub block_k: u32,
}

fn tg_per_core_tenths_for(
    steps: &[(u32, u32)],
    kv_length: u32,
) -> u32 {
    steps.iter().rev().find(|(minimum_kv, _)| kv_length >= *minimum_kv).map_or(steps[0].1, |(_, tenths)| *tenths)
}

fn splits_for_ratio(
    tg_per_core_tenths: u32,
    gpu_core_count: u32,
    threadgroups_per_kv_split: u32,
) -> u32 {
    (tg_per_core_tenths * gpu_core_count.max(1)).div_ceil(TENTHS_PER_RATIO * threadgroups_per_kv_split.max(1))
}

fn tabled_splits(
    head_dim: u32,
    suffix_length: u32,
    kv_length: u32,
    threadgroups_per_kv_split: u32,
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
            splits_for_ratio(tg_per_core_tenths_for(steps, kv_length), gpu_core_count, threadgroups_per_kv_split)
        })
}

pub fn choose_splits(
    geometry: SplitGeometry,
    suffix_length: u32,
    kv_length: u32,
    gpu_core_count: u32,
) -> u32 {
    let threadgroups_per_kv_split = (geometry.num_q_heads / geometry.num_groups * suffix_length)
        .div_ceil(geometry.block_rows)
        * geometry.num_groups;
    let max_splits = kv_length.div_ceil(geometry.block_k).max(1);

    if suffix_length > GEMM_GROUPED_DECODE_SUFFIX_MAX && kv_length >= PREFILL_SPLIT_MIN_KV {
        // Each padded row has head_dim partial values, one max, and one sum, all F32.
        let bytes_per_split = threadgroups_per_kv_split as u64
            * geometry.block_rows as u64
            * (geometry.head_dim as u64 + PARTIAL_STATS_PER_ROW)
            * DataType::F32.size_in_bytes() as u64;
        let requested_splits = kv_length.div_ceil(PREFILL_SPLIT_KEYS).min(max_splits);
        let scratch_splits = PREFILL_SPLIT_SCRATCH_BYTES / bytes_per_split;
        let splits = (requested_splits as u64).min(scratch_splits) as u32;
        if splits >= PREFILL_MIN_SPLITS {
            return splits;
        }
    }

    let measured_splits =
        tabled_splits(geometry.head_dim, suffix_length, kv_length, threadgroups_per_kv_split, gpu_core_count);
    let splits = match measured_splits {
        Some(splits) => splits,
        None => splits_for_ratio(TENTHS_PER_RATIO * TG_PER_CORE_FALLBACK, gpu_core_count, threadgroups_per_kv_split),
    };
    splits.clamp(1, max_splits)
}

#[cfg(test)]
#[path = "../../../../../../unit/backends/metal/kernel/attention/gemm_grouped_policy_test.rs"]
mod tests;
