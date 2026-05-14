#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GemmSpecializationError {
    ZeroTileDimension,
    ThreadgroupKExceedsGroupSize {
        threadgroup_k: u32,
        group_size: u32,
    },
}
