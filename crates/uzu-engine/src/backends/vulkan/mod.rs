mod error;
mod vk_buffer;
mod vk_buffer_error;
mod vk_command_buffer_completed;
mod vk_command_buffer_encoding;
mod vk_command_buffer_executable;
mod vk_command_buffer_pending;
mod vk_command_buffer_resources;
mod vk_compute_pipeline;
mod vk_context;
mod vk_context_create_info;
mod vk_context_error;
mod vk_kernels;
mod vk_logger;
mod vk_physical_device;
mod vk_physical_device_features;
mod vk_physical_device_subgroup_properties;
mod vk_println_logger;
mod vk_shader;
mod vk_timestamp_query_pool;

pub use error::Error;
pub use vk_buffer::VkBuffer;
pub use vk_buffer_error::VkBufferError;
pub use vk_command_buffer_completed::VkCommandBufferCompleted;
pub use vk_command_buffer_encoding::VkCommandBufferEncoding;
pub use vk_command_buffer_executable::VkCommandBufferExecutable;
pub use vk_command_buffer_pending::VkCommandBufferPending;
pub use vk_command_buffer_resources::VkCommandBufferResources;
pub use vk_compute_pipeline::VkComputePipeline;
pub use vk_context::VkContext;
pub use vk_context_create_info::VkContextCreateInfo;
pub use vk_context_error::VkContextError;
pub use vk_kernels::{
    A8QuantizedGemmVulkanKernel, A8QuantizedGemvVulkanKernel, ActivationTransformVulkanKernel, ActivationVulkanKernel,
    AncestorAttentionVulkanKernel, AttentionPrepareVulkanKernel, AttentionSinglePassVulkanKernel,
    BuildTreeGramVulkanKernel, BuildTreeOutVulkanKernel, BuildTreePrefixVulkanKernel, Conv1dDecodeVulkanKernel,
    Conv1dPackVulkanKernel, Conv1dScanVulkanKernel, ConvTreeScanVulkanKernel, DeltaNetConvScanVulkanKernel,
    DeltaNetConvUpdateVulkanKernel, DeltaNetNormGateVulkanKernel, DeltaNetPrefillPrepVulkanKernel,
    DeltaNetPrefillVulkanKernel, DeltaNetUpdateVulkanKernel, GatedActMulVulkanKernel, GemmVulkanKernel,
    GemvVulkanKernel, InputEmbeddingLookupVulkanKernel, KVCacheUpdateVulkanKernel, LogitTransformVulkanKernel,
    NormalizationVulkanKernel, PoolingMeanVulkanKernel, QKVNormVulkanKernel, QuantizedGemmVulkanKernel,
    QuantizedGemvVulkanKernel, SSDPrefill64VulkanKernel, SSDPrefillVulkanKernel, SSDUpdateVulkanKernel,
    SeparableCausalConvVulkanKernel, ShortConvDecodeVulkanKernel, ShortConvPackVulkanKernel,
    ShortConvPrefillVulkanKernel, ShortConvTrieVulkanKernel, SigmoidGateVulkanKernel, SoftmaxVulkanKernel,
    SplitInProjVulkanKernel, StateAdvanceVulkanKernel, TensorAddBiasVulkanKernel, TensorAddScaleVulkanKernel,
    TreeUpdateSolveVulkanKernel,
};
pub use vk_logger::VkLogger;
pub use vk_physical_device::VkPhysicalDevice;
pub use vk_physical_device_features::VkPhysicalDeviceFeatures;
pub use vk_physical_device_subgroup_properties::VkPhysicalDeviceSubgroupProperties;
pub use vk_println_logger::VkPrintlnLogger;
pub use vk_shader::VkShader;
pub use vk_timestamp_query_pool::VkTimestampQueryPool;

#[cfg(test)]
#[path = "../../../unit/backends/vulkan/mod.rs"]
mod tests;
