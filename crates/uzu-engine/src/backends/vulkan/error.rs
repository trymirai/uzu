#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("operation is not supported")]
    NotSupported,
    #[error(transparent)]
    Vulkan(#[from] ash::vk::Result),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    EntryPoint(#[from] std::ffi::NulError),
    #[error(transparent)]
    Context(#[from] super::VkContextError),
    #[error(transparent)]
    Buffer(#[from] super::VkBufferError),
    #[error("buffers must not be empty")]
    EmptyBuffer,
    #[error("buffer range {start}..{end} is empty or exceeds a buffer of {size} bytes")]
    BufferRange {
        start: u64,
        end: u64,
        size: u64,
    },
    #[error("copy ranges overlap within the same buffer")]
    CopyOverlap,
    #[error("fill range {start}..{end} must be 4-byte aligned")]
    FillAlignment {
        start: u64,
        end: u64,
    },
    #[error("kernel {kernel} has no variant for {data_types:?}")]
    KernelVariant {
        kernel: &'static str,
        data_types: Box<[crate::data_type::DataType]>,
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
    #[error("dispatch of {groups:?} groups exceeds the device limit {limit:?}")]
    DispatchGroups {
        groups: [u32; 3],
        limit: [u32; 3],
    },
}
