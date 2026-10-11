mod error;
mod vk_attention_kernel;
mod vk_buffer;
mod vk_buffer_error;
mod vk_command_buffer;
mod vk_command_buffer_completed;
mod vk_command_buffer_encoding;
mod vk_command_buffer_executable;
mod vk_command_buffer_pending;
mod vk_command_buffer_resources;
mod vk_compute_pipeline;
mod vk_context;
mod vk_context_create_info;
mod vk_context_error;
mod vk_delta_net_chunked_prefill;
mod vk_delta_net_tree_verify;
#[macro_use]
mod vk_kernels;
mod vk_logger;
mod vk_matmul_kernel;
mod vk_physical_device;
mod vk_physical_device_features;
mod vk_physical_device_subgroup_properties;
mod vk_println_logger;
mod vk_radix_top_k_small;
mod vk_shader;
mod vk_sparse_buffer;
mod vk_ssd_prefill_kernel;
mod vk_timestamp_query_pool;
mod vulkan;
mod vulkan_kernels;

pub use error::Error;
pub use vk_attention_kernel::VkAttentionKernel;
pub use vk_buffer::VkBuffer;
pub use vk_buffer_error::VkBufferError;
pub use vk_command_buffer::VkCommandBuffer;
pub use vk_command_buffer_completed::VkCommandBufferCompleted;
pub use vk_command_buffer_encoding::VkCommandBufferEncoding;
pub use vk_command_buffer_executable::VkCommandBufferExecutable;
pub use vk_command_buffer_pending::VkCommandBufferPending;
pub use vk_command_buffer_resources::VkCommandBufferResources;
pub use vk_compute_pipeline::VkComputePipeline;
pub use vk_context::VkContext;
pub use vk_context_create_info::VkContextCreateInfo;
pub use vk_context_error::VkContextError;
pub use vk_delta_net_chunked_prefill::VkDeltaNetChunkedPrefill;
pub use vk_delta_net_tree_verify::VkDeltaNetTreeVerify;
pub use vk_kernels::{
    A8QuantizedGemmVulkanKernel, A8QuantizedGemvVulkanKernel, ActivationTransformVulkanKernel, ActivationVulkanKernel,
    AncestorAttentionVulkanKernel, AttentionPrepareVulkanKernel, AttentionSinglePassVulkanKernel,
    BuildTreeGramVulkanKernel, BuildTreeOutVulkanKernel, BuildTreePrefixVulkanKernel, ContextRingUpdateVulkanKernel,
    Conv1dDecodeVulkanKernel, Conv1dPackVulkanKernel, Conv1dScanVulkanKernel, ConvTreeScanVulkanKernel,
    DeltaNetChunkedADiagInvVulkanKernel, DeltaNetChunkedCausalInvVulkanKernel, DeltaNetChunkedCumsumVulkanKernel,
    DeltaNetChunkedGramVulkanKernel, DeltaNetChunkedOutputAndStateVulkanKernel, DeltaNetConvScanVulkanKernel,
    DeltaNetConvUpdateVulkanKernel, DeltaNetNormGateVulkanKernel, DeltaNetPrefillPrepVulkanKernel,
    DeltaNetPrefillVulkanKernel, DeltaNetUpdateVulkanKernel, GatedActMulVulkanKernel, GemmVulkanKernel,
    GemvVulkanKernel, InputEmbeddingLookupVulkanKernel, KVCacheUpdateVulkanKernel, LogitTransformVulkanKernel,
    NormalizationVulkanKernel, PoolingMeanVulkanKernel, QKVNormVulkanKernel, QuantizedGemmVulkanKernel,
    QuantizedGemvVulkanKernel, RadixTopKSmallVulkanKernel, RepetitionPenaltyVulkanKernel, SSDPrefill64VulkanKernel,
    SSDPrefillVulkanKernel, SSDUpdateVulkanKernel, SeparableCausalConvVulkanKernel, ShortConvDecodeVulkanKernel,
    ShortConvPackVulkanKernel, ShortConvPrefillVulkanKernel, ShortConvTrieVulkanKernel, SigmoidGateVulkanKernel,
    SoftmaxVulkanKernel, SplitInProjVulkanKernel, StateAdvanceVulkanKernel, StridedGemmVulkanKernel,
    StridedGemvVulkanKernel, TensorAddBiasVulkanKernel, TensorAddScaleVulkanKernel, TreeUpdateSolveVulkanKernel,
    TrellisGemvVulkanKernel, TrellisTransformVulkanKernel, UnifiedSamplingVulkanKernel,
    WeaverFrontierInsertChildrenVulkanKernel, WeaverFrontierSelectVulkanKernel, WeaverTopChildrenVulkanKernel,
};
pub use vk_logger::VkLogger;
pub use vk_matmul_kernel::VkMatmulKernel;
pub use vk_physical_device::VkPhysicalDevice;
pub use vk_physical_device_features::VkPhysicalDeviceFeatures;
pub use vk_physical_device_subgroup_properties::VkPhysicalDeviceSubgroupProperties;
pub use vk_println_logger::VkPrintlnLogger;
pub use vk_radix_top_k_small::VkRadixTopKSmall;
pub use vk_shader::VkShader;
pub use vk_sparse_buffer::VkSparseBuffer;
pub use vk_ssd_prefill_kernel::VkSSDPrefillKernel;
pub use vk_timestamp_query_pool::VkTimestampQueryPool;
pub use vulkan::Vulkan;
pub use vulkan_kernels::VulkanKernels;

#[cfg(test)]
#[path = "../../../unit/backends/vulkan/mod.rs"]
mod tests;
