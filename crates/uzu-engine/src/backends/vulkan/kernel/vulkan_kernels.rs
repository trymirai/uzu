use crate::backends::{
    common::{
        Kernels,
        kernel::{Unsupported, tree_verify_blocks::TreeVerifyBlocks},
    },
    vulkan::{
        VkAttentionKernel, VkDeltaNetChunkedPrefill, VkDeltaNetUpdateKernel, VkLinearizeChain, VkLinearizeTree,
        VkMatmulKernel, VkRadixTopKSmall, VkTreeVerify, Vulkan,
    },
};

include!(concat!(env!("OUT_DIR"), "/vulkan.rs"));

pub struct VulkanKernels;

impl Kernels for VulkanKernels {
    type Backend = Vulkan;

    autogen_kernels!();
    type AttentionKernel = VkAttentionKernel;
    type IndexerKernel = Unsupported<Vulkan>;
    type DeltaNetChunkedPrefill = VkDeltaNetChunkedPrefill;
    type DeltaNetTreeVerify = TreeVerifyBlocks<Vulkan>;
    type DeltaNetUpdateKernel = VkDeltaNetUpdateKernel;
    type LinearizeChainKernel = VkLinearizeChain;
    type LinearizeTreeKernel = VkLinearizeTree;
    type MatmulKernel = VkMatmulKernel;
    type RadixTopKSmall = VkRadixTopKSmall;
    type TreeVerifyKernel = VkTreeVerify;
}
