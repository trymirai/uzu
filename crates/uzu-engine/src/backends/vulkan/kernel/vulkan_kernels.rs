use crate::backends::{
    common::Kernels,
    vulkan::{
        VkAttentionKernel, VkDeltaNetChunkedPrefill, VkDeltaNetTreeVerify, VkDeltaNetUpdateKernel, VkMatmulKernel,
        VkRadixTopKSmall, VkTreeVerify, Vulkan,
    },
};

include!(concat!(env!("OUT_DIR"), "/vulkan.rs"));

pub struct VulkanKernels;

impl Kernels for VulkanKernels {
    type Backend = Vulkan;

    autogen_kernels!();
    type AttentionKernel = VkAttentionKernel;
    type DeltaNetChunkedPrefill = VkDeltaNetChunkedPrefill;
    type DeltaNetTreeVerify = VkDeltaNetTreeVerify;
    type DeltaNetUpdateKernel = VkDeltaNetUpdateKernel;
    type MatmulKernel = VkMatmulKernel;
    type RadixTopKSmall = VkRadixTopKSmall;
    type TreeVerifyKernel = VkTreeVerify;
}
