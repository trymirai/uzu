use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum GemmSpecializationError {
    #[error("threadgroup K={threadgroup_k} exceeds group size {group_size}")]
    ThreadgroupKExceedsGroupSize {
        threadgroup_k: u32,
        group_size: u32,
    },
    #[error("quantized B requires transposed layout")]
    QuantizedRequiresTransposedB,
}
