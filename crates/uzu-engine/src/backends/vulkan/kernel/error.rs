use crate::data_type::DataType;

#[derive(Debug, thiserror::Error)]
pub enum VkKernelError {
    #[error("kernel {kernel} has no variant for {data_types:?}")]
    KernelVariant {
        kernel: &'static str,
        data_types: Box<[DataType]>,
    },
    #[error("kernel {kernel} requires {condition}")]
    KernelPrecondition {
        kernel: &'static str,
        condition: &'static str,
    },
    #[error("work group {size:?} exceeds the device limit {limit:?} or {invocations} invocations")]
    WorkGroupSize {
        size: [u32; 3],
        limit: [u32; 3],
        invocations: u32,
    },
}
