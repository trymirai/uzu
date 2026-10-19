pub const COLUMN_GROUP_COUNT: u32 = 4;

/// Mixing matrix side length: columns divided by their largest power-of-two factor.
pub fn mixing_dimension(columns: u32) -> u32 {
    columns >> columns.trailing_zeros()
}

/// Rows of a trellis activation buffer: the trellis GEMM loads whole M tiles (at most 128 rows) without row bounds
/// checks; rows past the batch are never stored.
pub fn padded_activation_rows(batch_dim: u32) -> u32 {
    batch_dim.next_multiple_of(128)
}

/// Widest row the trellis embedding lookup can unrotate in threadgroup memory (32 KiB of f32).
pub const LOOKUP_MAX_COLUMNS: u32 = 8192;
