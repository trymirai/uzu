use crate::backends::vulkan::{AttentionGemmVulkanKernel, AttentionSinglePassVulkanKernel};

pub enum VkAttentionManyQueries {
    Gemm([AttentionGemmVulkanKernel; 2]),
    SinglePass([AttentionSinglePassVulkanKernel; 2]),
}
