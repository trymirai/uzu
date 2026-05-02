#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnifiedGemmSpecializationError {
    ZeroTileDimension,
    ZeroGroupSize,
    UnsupportedBitsPerWeight(u8),
    ThreadgroupKExceedsGroupSize {
        threadgroup_k: u32,
        group_size: u32,
    },
}
