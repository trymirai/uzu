#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmSpecializationError {
    ThreadgroupKExceedsGroupSize {
        threadgroup_k: u32,
        group_size: u32,
    },
    QuantizedRequiresSimdgroup,
    QuantizedRequiresTransposedB,
    QuantizedRoutedThroughFpKernel,
}
