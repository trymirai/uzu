pub const COLUMN_GROUP_COUNT: u32 = 4;

/// Mixing matrix side length: columns divided by their largest power-of-two factor.
pub fn mixing_dimension(columns: u32) -> u32 {
    columns >> columns.trailing_zeros()
}

/// Widest row the trellis embedding lookup can unrotate in threadgroup memory (32 KiB of f32).
pub const LOOKUP_MAX_COLUMNS: u32 = 8192;
