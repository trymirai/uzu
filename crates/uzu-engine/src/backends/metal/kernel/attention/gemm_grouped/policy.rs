use super::MaskKind;
use crate::{
    backends::common::{gpu_types::kv_layout, kernel::AttentionKernelConfig},
    data_type::DataType,
};

pub const MAX_TRIE_SUFFIX: u32 = 64;

const MXU_DECODE_MAX_SUFFIX: u32 = 64;
const SHORT_SUFFIX_MAX: u32 = 4;

pub fn is_supported(arguments: &AttentionKernelConfig) -> bool {
    const SUPPORTED_HEAD_DIMS: [u32; 2] = [128, 256];

    SUPPORTED_HEAD_DIMS.contains(&arguments.head_dim)
        && arguments.data_type == DataType::BF16
        && !arguments.has_sinks
        && !arguments.is_kv_cache_ring
        && arguments.sliding_window_size.is_none()
        && arguments.scale.is_none_or(|scale| scale > 0.0)
        && arguments.num_groups > 0
        && arguments.num_q_heads > 0
        && arguments.num_q_heads.is_multiple_of(arguments.num_groups)
}

pub fn select_head_dim_slices(
    head_dim: u32,
    mask: MaskKind,
    suffix_length: u32,
    kv_length: u32,
    use_mxu: bool,
) -> Option<u32> {
    if use_mxu {
        select_mxu_head_dim_slices(head_dim, mask, suffix_length)
    } else {
        select_simdgroup_head_dim_slices(head_dim, mask, suffix_length, kv_length)
    }
}

fn select_mxu_head_dim_slices(
    head_dim: u32,
    mask: MaskKind,
    suffix_length: u32,
) -> Option<u32> {
    const MXU_DECODE_MIN_SUFFIX: u32 = 2;
    const MXU_PREFILL_MAX_SUFFIX: u32 = 1024;

    let decode = (MXU_DECODE_MIN_SUFFIX..=MXU_DECODE_MAX_SUFFIX).contains(&suffix_length);
    let causal_prefill = head_dim == 256
        && mask == MaskKind::Causal
        && (MXU_DECODE_MAX_SUFFIX + 1..=MXU_PREFILL_MAX_SUFFIX).contains(&suffix_length);
    (decode || causal_prefill).then_some(head_dim / kv_layout::SLICE_DIMS)
}

fn select_simdgroup_head_dim_slices(
    head_dim: u32,
    mask: MaskKind,
    suffix_length: u32,
    kv_length: u32,
) -> Option<u32> {
    const SIMDGROUP_DECODE_MIN_KV_LENGTH: u32 = 16_384;
    const SIMDGROUP_PREFILL_MIN_KV_LENGTH: u32 = 61_440;

    match (mask, head_dim, suffix_length) {
        (MaskKind::Causal, 256, 1) if kv_length >= SIMDGROUP_DECODE_MIN_KV_LENGTH => Some(8),
        (MaskKind::Causal, 256, 2) if kv_length >= SIMDGROUP_DECODE_MIN_KV_LENGTH => Some(4),
        (MaskKind::Causal, 256, 4) if kv_length >= SIMDGROUP_DECODE_MIN_KV_LENGTH => Some(2),
        (MaskKind::Causal, 128, 1 | 2 | 4) if kv_length >= SIMDGROUP_DECODE_MIN_KV_LENGTH => Some(4),
        (MaskKind::Causal, 128, 512 | 1024) if kv_length >= 30_720 => Some(1),
        (MaskKind::Causal, 256, 512 | 1024) if kv_length >= SIMDGROUP_PREFILL_MIN_KV_LENGTH => Some(2),
        _ => None,
    }
}

type MeasuredSplits = (u32, u32, &'static [(u32, u32)]);

const MEASURED_SUFFIX_MIN: u32 = 2;
const MEASURED_SUFFIX_MAX: u32 = 64;

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
const PARTIAL_STATS_PER_ROW: u64 = 2;

fn partial_row_bytes(head_dim: u32) -> u64 {
    head_dim as u64 * DataType::BF16.size_in_bytes() as u64
        + PARTIAL_STATS_PER_ROW * DataType::F32.size_in_bytes() as u64
}

pub struct SplitGeometry {
    pub head_dim: u32,
    pub num_q_heads: u32,
    pub num_groups: u32,
    pub query_rows_per_threadgroup: u32,
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
    use_mxu: bool,
) -> u32 {
    let threadgroups_per_kv_split = (geometry.num_q_heads / geometry.num_groups * suffix_length)
        .div_ceil(geometry.query_rows_per_threadgroup)
        * geometry.num_groups;
    let max_splits = kv_length.div_ceil(geometry.block_k).max(1);

    if !use_mxu {
        const MAX_SPLITS: u32 = 16;
        let threadgroups_per_split = threadgroups_per_kv_split.max(1);
        let splits = if suffix_length > SHORT_SUFFIX_MAX {
            let bytes_per_split = threadgroups_per_split as u64
                * geometry.query_rows_per_threadgroup as u64
                * partial_row_bytes(geometry.head_dim);
            (PREFILL_SPLIT_SCRATCH_BYTES / bytes_per_split) as u32
        } else {
            (gpu_core_count.max(1) * 4).div_ceil(threadgroups_per_split)
        };
        return splits.clamp(1, max_splits.min(MAX_SPLITS));
    }

    if suffix_length > MXU_DECODE_MAX_SUFFIX && kv_length >= PREFILL_SPLIT_MIN_KV {
        let f32_estimated_bytes_per_split = threadgroups_per_kv_split as u64
            * geometry.query_rows_per_threadgroup as u64
            * (geometry.head_dim as u64 + PARTIAL_STATS_PER_ROW)
            * DataType::F32.size_in_bytes() as u64;
        let requested_splits = kv_length.div_ceil(PREFILL_SPLIT_KEYS).min(max_splits);
        let scratch_splits = PREFILL_SPLIT_SCRATCH_BYTES / f32_estimated_bytes_per_split;
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

pub fn int8_splits(
    split_rows: u32,
    head_dim: u32,
    suffix_length: u32,
    kv_length: u32,
) -> u32 {
    const SHORT_SUFFIX_MAX_SPLITS: u32 = 16;
    let kv_blocks = kv_length.div_ceil(kv_layout::TILE_KEYS);
    let splits = if suffix_length <= MXU_DECODE_MAX_SUFFIX {
        let query_tokens = suffix_length.max(kv_layout::TILE_QUERIES);
        let splits = (kv_length as f32 / query_tokens as f32).sqrt().round() as u32;
        if suffix_length <= SHORT_SUFFIX_MAX {
            splits.min(SHORT_SUFFIX_MAX_SPLITS)
        } else {
            splits
        }
    } else if kv_length >= PREFILL_SPLIT_MIN_KV {
        let bytes_per_split = split_rows as u64 * partial_row_bytes(head_dim);
        let scratch_splits = (PREFILL_SPLIT_SCRATCH_BYTES / bytes_per_split) as u32;
        let rounded_splits = (kv_length + PREFILL_SPLIT_KEYS / 2) / PREFILL_SPLIT_KEYS;
        rounded_splits.min(scratch_splits)
    } else {
        1
    };
    splits.clamp(1, kv_blocks.max(1))
}

#[cfg(test)]
#[path = "../../../../../../unit/backends/metal/kernel/attention/gemm_grouped_policy_test.rs"]
mod tests;
