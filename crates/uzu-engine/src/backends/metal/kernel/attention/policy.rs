use crate::{
    backends::common::{gpu_types::kv_layout, kernel::AttentionKernelConfig},
    data_type::DataType,
};

pub const MAX_TRIE_SUFFIX: u32 = 64;
const MXU_DECODE_MAX_SUFFIX: u32 = 64;
const SHORT_SUFFIX_MAX: u32 = 4;
const PREFILL_SPLIT_KEYS: u32 = 2048;
const PREFILL_SPLIT_MIN_KV: u32 = 16384;
const PREFILL_SPLIT_SCRATCH_BYTES: u64 = 256 << 20;

fn partial_row_bytes(head_dim: u32) -> u64 {
    head_dim as u64 * DataType::BF16.size_in_bytes() as u64 + 2 * DataType::F32.size_in_bytes() as u64
}

pub fn supports_mxu(arguments: &AttentionKernelConfig) -> bool {
    matches!(arguments.head_dim, 128 | 256)
        && arguments.data_type == DataType::BF16
        && !arguments.has_sinks
        && !arguments.is_kv_cache_ring
        && arguments.sliding_window_size.is_none()
        && arguments.scale.is_none_or(|scale| scale > 0.0)
        && arguments.num_groups > 0
        && arguments.num_q_heads > 0
        && arguments.num_q_heads.is_multiple_of(arguments.num_groups)
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

pub fn int8_simd_splits(
    split_rows: u32,
    head_dim: u32,
    suffix_length: u32,
    kv_length: u32,
    threadgroups_per_split: u32,
    simdgroups_per_threadgroup: u32,
    gpu_core_count: u32,
) -> u32 {
    if suffix_length <= MXU_DECODE_MAX_SUFFIX {
        // Small query tiles need more threadgroups to hide memory latency.
        // Keep at least eight SIMD groups per core, with two resident groups
        // for the larger tiles. Partition only at packed KV block boundaries.
        let groups_per_core = 8_u32.div_ceil(simdgroups_per_threadgroup).max(2);
        let occupancy_splits = (gpu_core_count.max(1) * groups_per_core).div_ceil(threadgroups_per_split.max(1));
        occupancy_splits.clamp(1, kv_length.div_ceil(kv_layout::TILE_KEYS).max(1))
    } else {
        int8_splits(split_rows, head_dim, suffix_length, kv_length)
    }
}

#[cfg(test)]
#[path = "../../../../../unit/backends/metal/kernel/attention/int8_policy_test.rs"]
mod tests;
