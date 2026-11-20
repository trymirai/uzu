use crate::{
    backends::metal::gpu_types::{attention::AttnParams, kv_layout},
    data_type::DataType,
};

const MXU_DECODE_MAX_SUFFIX: u32 = 64;
const SHORT_SUFFIX_MAX: u32 = 4;
// Short decodes are latency-bound: split to twice the threadgroups that fill the GPU.
const SHORT_SUFFIX_FILL_MULTIPLE: u32 = 2;
const PREFILL_SPLIT_KEYS: u32 = 2048;
const PREFILL_SPLIT_MIN_KV: u32 = 16384;
const PREFILL_SPLIT_SCRATCH_BYTES: u64 = 256 << 20;
const SPARSE_THREADGROUPS_PER_CORE: u32 = 4;

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

pub fn splits(
    params: &AttnParams,
    head_dim: u32,
    split_rows: u32,
    fill_splits: u32,
    use_mxu: bool,
) -> u32 {
    let splits = if params.q_len <= MXU_DECODE_MAX_SUFFIX {
        if use_mxu {
            let query_tokens = params.q_len.max(kv_layout::TILE_QUERIES);
            let splits = (params.k_len as f32 / query_tokens as f32).sqrt().round() as u32;
            if params.q_len <= SHORT_SUFFIX_MAX {
                splits.min(fill_splits * SHORT_SUFFIX_FILL_MULTIPLE)
            } else {
                splits
            }
        } else {
            fill_splits
        }
    } else if params.k_len >= PREFILL_SPLIT_MIN_KV {
        let row_bytes =
            head_dim as u64 * DataType::BF16.size_in_bytes() as u64 + 2 * DataType::F32.size_in_bytes() as u64;
        let bytes_per_split = split_rows as u64 * row_bytes;
        let scratch_splits = (PREFILL_SPLIT_SCRATCH_BYTES / bytes_per_split) as u32;
        let rounded_splits = (params.k_len + PREFILL_SPLIT_KEYS / 2) / PREFILL_SPLIT_KEYS;
        rounded_splits.min(scratch_splits)
    } else {
        1
    };
    splits.clamp(1, params.nk.max(1))
}
