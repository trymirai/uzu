use crate::{
    backends::metal::gpu_types::{attention::AttnParams, kv_layout},
    data_type::DataType,
};

const MXU_DECODE_MAX_SUFFIX: u32 = 64;
const SHORT_SUFFIX_MAX: u32 = 4;
const PREFILL_SPLIT_KEYS: u32 = 2048;
const PREFILL_SPLIT_MIN_KV: u32 = 16384;
const PREFILL_SPLIT_SCRATCH_BYTES: u64 = 256 << 20;

pub fn splits(
    params: &AttnParams,
    head_dim: u32,
    split_rows: u32,
    occupancy: Option<u32>,
) -> u32 {
    const SHORT_SUFFIX_MAX_SPLITS: u32 = 16;
    let splits = if params.q_len <= MXU_DECODE_MAX_SUFFIX {
        occupancy.unwrap_or_else(|| {
            let query_tokens = params.q_len.max(kv_layout::TILE_QUERIES);
            let splits = (params.k_len as f32 / query_tokens as f32).sqrt().round() as u32;
            if params.q_len <= SHORT_SUFFIX_MAX {
                splits.min(SHORT_SUFFIX_MAX_SPLITS)
            } else {
                splits
            }
        })
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
