/// Mixing matrix side length: columns divided by their largest power-of-two factor
pub fn mixing_dimension(columns: u32) -> u32 {
    columns >> columns.trailing_zeros()
}
