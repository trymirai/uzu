#[repr(C)]
#[allow(non_snake_case)]
#[derive(Debug, Default, Copy, Clone)]
pub struct GemmParams {
    pub M: u32,
    pub N: u32,
    pub K: u32,
    pub leading_dimension_a: u32,
    pub leading_dimension_b: u32,
    pub leading_dimension_d: u32,
    pub threadgroups_per_column: u32,
    pub threadgroups_per_row: u32,
    pub aligned_inner_iterations: u32,
    pub use_morton: bool,
    pub ab_scale: f32,
}

/// Everything the trellis decode needs that is not already a `GemmParams` field.
///
/// `l` and `k_bits` are runtime scalars rather than kernel variants on purpose:
/// they only pick a mask and a stride, so specializing on them would multiply
/// PSOs for nothing. `hash_a`/`hash_b` are `trellis_format::hash_params()`,
/// passed as scalars so the kernel needs no codebook buffer at all. Build one
/// with `trellis_format::trellis_params`, which is the only place these are
/// derived.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct TrellisParams {
    /// Trellis window width in bits, `1 ..= 32`.
    pub l: u32,
    /// Bits injected per trellis step, i.e. `k * V`.
    pub k_bits_per_step: u32,
    /// `u32`s between the starts of two consecutive tape rows.
    pub row_stride_words: u32,
    pub hash_a: u32,
    pub hash_b: u32,
    /// `1 / rms(codebook)`, folded into the per-row scale in the epilogue.
    pub codebook_scale: f32,
}
