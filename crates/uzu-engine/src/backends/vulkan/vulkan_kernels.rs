use super::{
    VkAttentionKernel, VkDeltaNetChunkedPrefill, VkDeltaNetTreeVerify, VkMatmulKernel, VkRadixTopKSmall,
    VkSSDPrefillKernel, Vulkan,
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
    type MatmulKernel = VkMatmulKernel;
    type RadixTopKSmall = VkRadixTopKSmall;
}
