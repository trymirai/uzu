use crate::backends::vulkan::{AttentionGemmVulkanKernel, AttentionSinglePassVulkanKernel};

/// The kernels for more queries than the two-pass kernels take, without and with a trie: AttentionGemm for 64, 128 or
/// 256 dimensions on devices with its workgroup memory, else AttentionSinglePass.
pub enum VkAttentionManyQueries {
    Gemm([AttentionGemmVulkanKernel; 2]),
    SinglePass([AttentionSinglePassVulkanKernel; 2]),
}
