use std::cmp::Reverse;

use crate::{
    backends::metal::gpu_types::{attention::AttnParams, kv_layout},
    data_type::DataType,
};

const MXU_DECODE_MAX_SUFFIX: u32 = 64;
// Measured best: more rows cost occupancy.
const PREFILL_TILE_ROWS: u32 = 2 * kv_layout::TILE_QUERIES;
const DECODE_THREADGROUPS_PER_CORE: u32 = 4;
// Fill the GPU only if it at least doubles the splits.
const FILL_SPLITS_MIN_GAIN: u32 = 2;
const PREFILL_SPLIT_KEYS: u32 = 2048;
const PREFILL_SPLIT_MIN_KV: u32 = 16384;
const PREFILL_SPLIT_SCRATCH_BYTES: u64 = 256 << 20;
const SPARSE_THREADGROUPS_PER_CORE: u32 = 4;

// Simd decode, measured on M2/M4.
const SIMD_SIMDGROUPS_PER_CORE: u32 = 8;
const SIMD_MIN_THREADGROUPS_PER_CORE: u32 = 2;
const SIMD_KEY_SPLIT_MAX_SIMDGROUPS: u32 = 4;
const SIMD_KEY_SPLIT_MAX_FILLS: u32 = 8;
const SIMD_KEYS_PER_SPLIT: u32 = 150;

pub fn sparse_splits(
    gpu_cores: u32,
    rows: u32,
    blocks_per_query: u32,
    compression_ratio: u32,
) -> u32 {
    (SPARSE_THREADGROUPS_PER_CORE * gpu_cores)
        .div_ceil(rows)
        .min((blocks_per_query * compression_ratio).div_ceil(kv_layout::TILE_KEYS))
        .max(1)
}

pub fn tile_order(
    rows: u32,
    grouped_rows: u32,
    suffix_length: u32,
    use_mxu: bool,
) -> (u32, Reverse<u32>) {
    if use_mxu && suffix_length > MXU_DECODE_MAX_SUFFIX {
        (rows.abs_diff(PREFILL_TILE_ROWS), Reverse(0))
    } else if use_mxu {
        (grouped_rows.div_ceil(rows) * rows, Reverse(rows))
    } else {
        // Every threadgroup reads all of K and V, so fewer threadgroups read less.
        let threadgroups = grouped_rows.div_ceil(rows);
        (threadgroups, Reverse(0))
    }
}

pub fn splits(
    params: &AttnParams,
    head_dim: u32,
    split_rows: u32,
    gpu_cores: u32,
    threadgroups: u32,
    useful_simdgroups: Option<u32>,
    use_mxu: bool,
    sliding_window_size: Option<u32>,
) -> u32 {
    let threadgroups = threadgroups.max(1);
    // Simd kernels walk only the window and the suffix.
    let keys = if let Some(window) = sliding_window_size
        && !use_mxu
    {
        window.saturating_add(params.q_len).min(params.k_len)
    } else {
        params.k_len
    };
    let splits = if params.q_len > MXU_DECODE_MAX_SUFFIX {
        prefill_splits(params, keys, head_dim, split_rows)
    } else if use_mxu {
        mxu_decode_splits(params, head_dim, gpu_cores, threadgroups)
    } else {
        simd_decode_splits(keys, head_dim, gpu_cores, threadgroups, useful_simdgroups)
    };
    // Two more blocks for window ends that are not block aligned.
    let max_splits = keys.div_ceil(kv_layout::TILE_KEYS) + 2;
    splits.clamp(1, max_splits.min(params.nk).max(1))
}

fn prefill_splits(
    params: &AttnParams,
    keys: u32,
    head_dim: u32,
    split_rows: u32,
) -> u32 {
    if params.k_len < PREFILL_SPLIT_MIN_KV {
        return 1;
    }
    let row_bytes = head_dim as u64 * DataType::BF16.size_in_bytes() as u64 + 2 * DataType::F32.size_in_bytes() as u64;
    let scratch_splits = (PREFILL_SPLIT_SCRATCH_BYTES / (split_rows as u64 * row_bytes)) as u32;
    let rounded_splits = (keys + PREFILL_SPLIT_KEYS / 2) / PREFILL_SPLIT_KEYS;
    rounded_splits.min(scratch_splits)
}

fn mxu_decode_splits(
    params: &AttnParams,
    head_dim: u32,
    gpu_cores: u32,
    threadgroups: u32,
) -> u32 {
    let query_tokens = params.q_len.max(kv_layout::TILE_QUERIES);
    let splits = (params.k_len as f32 / query_tokens as f32).sqrt().round() as u32;
    let fill_splits = (gpu_cores * DECODE_THREADGROUPS_PER_CORE)
        .div_ceil(threadgroups)
        .min(params.nk.div_ceil(head_dim / kv_layout::SLICE_DIMS));
    if fill_splits >= FILL_SPLITS_MIN_GAIN * splits {
        fill_splits
    } else {
        splits
    }
}

fn simd_decode_splits(
    keys: u32,
    head_dim: u32,
    gpu_cores: u32,
    threadgroups: u32,
    useful_simdgroups: Option<u32>,
) -> u32 {
    let (threadgroups_per_core, split_by_keys) = if let Some(simdgroups) = useful_simdgroups {
        (
            (SIMD_SIMDGROUPS_PER_CORE / simdgroups).max(SIMD_MIN_THREADGROUPS_PER_CORE),
            head_dim <= 128 && simdgroups <= SIMD_KEY_SPLIT_MAX_SIMDGROUPS,
        )
    } else {
        // Vector kernel.
        (DECODE_THREADGROUPS_PER_CORE, head_dim == 128)
    };
    let fill_splits = |fills: u32| (fills * gpu_cores * threadgroups_per_core).div_ceil(threadgroups);
    if !split_by_keys {
        return fill_splits(1);
    }
    let key_splits = keys.next_multiple_of(kv_layout::TILE_KEYS) / SIMD_KEYS_PER_SPLIT;
    fill_splits(1).max(key_splits.min(fill_splits(SIMD_KEY_SPLIT_MAX_FILLS)))
}
