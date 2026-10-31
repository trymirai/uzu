use ash::vk;

#[derive(Debug, thiserror::Error)]
pub enum VkPipelineError {
    #[error("shader is not valid SPIR-V: {0}")]
    Spirv(#[source] std::io::Error),
    #[error("entry point name contains NUL: {0}")]
    EntryPoint(#[from] std::ffi::NulError),
    #[error("compute pipelines cannot require subgroups of {0} invocations on this device")]
    SubgroupSize(u32),
    #[error(transparent)]
    Vulkan(#[from] vk::Result),
}
