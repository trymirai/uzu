#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnifiedGemmSpecializationError {
    ZeroTileDimension,
    ZeroGroupSize,
    ThreadgroupKExceedsGroupSize {
        threadgroup_k: u32,
        group_size: u32,
    },
}
