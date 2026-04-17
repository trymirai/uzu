//! Attention kernel parameter structs.

/// Parameters for the GEMM attention kernel.
///
/// All strides are in **elements**, not bytes.
/// Q/K/V/O are treated as 3D tensors with strides for:
/// - `[0]` batch
/// - `[1]` head (or kv-head for K/V)
/// - `[2]` sequence (row stride for [seq, head_dim])
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct AttnParams {
    pub q_strides: [i64; 3],
    pub k_strides: [i64; 3],
    pub v_strides: [i64; 3],
    pub o_strides: [i64; 3],
    pub gqa_factor: i32,
    pub scale: f32,
    /// Query length (suffix length)
    pub q_len: i32,
    /// Key length (prefix + suffix)
    pub k_len: i32,
    /// Absolute offset of the first query token in the full key sequence.
    /// For LLM decode/prefill this is the segment prefix length.
    pub q_off: i32,
    /// Query tiling: q_len / BQ
    pub nq_aligned: i32,
    /// Query tiling: q_len % BQ
    pub q_rem: i32,
    /// Key tiling: ceil(k_len / BK)
    pub nk: i32,
    /// Key tiling: k_len / BK
    pub nk_aligned: i32,
    /// Key tiling: k_len % BK
    pub k_rem: i32,
}
