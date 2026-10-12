use super::{
    VkAttentionKernel, VkDeltaNetChunkedPrefill, VkDeltaNetTreeVerify, VkDeltaNetUpdateKernel, VkMatmulKernel,
    VkRadixTopKSmall, VkSSDPrefillKernel, Vulkan,
};
use crate::backends::common::Kernels;

pub struct VulkanKernels;

impl Kernels for VulkanKernels {
    type Backend = Vulkan;

    autogen_kernels!();
    type SSDPrefillKernel = VkSSDPrefillKernel;
    type AttentionKernel = VkAttentionKernel;
    type DeltaNetChunkedPrefill = VkDeltaNetChunkedPrefill;
    type DeltaNetTreeVerify = VkDeltaNetTreeVerify;
    type DeltaNetUpdateKernel = VkDeltaNetUpdateKernel;
    type MatmulKernel = VkMatmulKernel;
    type RadixTopKSmall = VkRadixTopKSmall;
}
