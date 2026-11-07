use crate::{
    backends::{
        common::{
            BufferMut, BufferRef, Kernels,
            kernel::{MoeCombineKernel, MoeGemvKernel, MoeRouterTopKKernel, Unsupported},
        },
        vulkan::{
            Error, VkAttentionKernel, VkCommandBufferEncoding, VkContext, VkDeltaNetChunkedPrefill,
            VkDeltaNetTreeVerify, VkDeltaNetUpdateKernel, VkMatmulKernel, VkRadixTopKSmall, VkTreeVerify, Vulkan,
        },
    },
    data_type::DataType,
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
    // No model the engine loads routes to experts yet, so Vulkan has no MoE kernels.
    type MoeCombineKernel = Unsupported<Vulkan>;
    type MoeGemvKernel = Unsupported<Vulkan>;
    type MoeRouterTopKKernel = Unsupported<Vulkan>;
    type RadixTopKSmall = VkRadixTopKSmall;
    type TreeVerifyKernel = VkTreeVerify;
}

impl MoeRouterTopKKernel for Unsupported<Vulkan> {
    type Backend = Vulkan;

    fn new(
        _context: &VkContext,
        _data_type: DataType,
        _max_top_k: u32,
    ) -> Result<Self, Error> {
        Err(Error::NotSupported)
    }

    fn encode(
        &self,
        _logits: impl BufferRef<Backend = Vulkan>,
        _expert_ids: impl BufferMut<Backend = Vulkan>,
        _expert_weights: impl BufferMut<Backend = Vulkan>,
        _num_experts: u32,
        _top_k: u32,
        _batch_size: u32,
        _command_buffer: &mut VkCommandBufferEncoding,
    ) {
        unreachable!("unsupported MoE router should not be constructed")
    }
}

impl MoeGemvKernel for Unsupported<Vulkan> {
    type Backend = Vulkan;

    fn new(
        _context: &VkContext,
        _data_type: DataType,
        _group_size: u32,
        _bits: u32,
        _a_per_token: bool,
    ) -> Result<Self, Error> {
        Err(Error::NotSupported)
    }

    fn encode(
        &self,
        _b: impl BufferRef<Backend = Vulkan>,
        _scales: impl BufferRef<Backend = Vulkan>,
        _zero_points: Option<impl BufferRef<Backend = Vulkan>>,
        _a: impl BufferRef<Backend = Vulkan>,
        _expert_ids: impl BufferRef<Backend = Vulkan>,
        _d: impl BufferMut<Backend = Vulkan>,
        _hadamard_factors: impl BufferRef<Backend = Vulkan>,
        _in_vec_size: u32,
        _out_vec_size: u32,
        _num_routes: u32,
        _active_experts_per_token: u32,
        _group_count_x: u32,
        _b_expert_stride_bytes: u32,
        _scale_expert_stride: u32,
        _zero_point_expert_stride: u32,
        _scale_output_stride: u32,
        _scale_group_stride: u32,
        _zero_point_output_stride: u32,
        _zero_point_group_stride: u32,
        _command_buffer: &mut VkCommandBufferEncoding,
    ) {
        unreachable!("unsupported MoE GEMV should not be constructed")
    }
}

impl MoeCombineKernel for Unsupported<Vulkan> {
    type Backend = Vulkan;

    fn new(
        _context: &VkContext,
        _data_type: DataType,
        _bd: u32,
    ) -> Result<Self, Error> {
        Err(Error::NotSupported)
    }

    fn encode(
        &self,
        _expert_outputs: impl BufferRef<Backend = Vulkan>,
        _expert_weights: impl BufferRef<Backend = Vulkan>,
        _shared_output: impl BufferRef<Backend = Vulkan>,
        _shared_gate: impl BufferRef<Backend = Vulkan>,
        _output: impl BufferMut<Backend = Vulkan>,
        _model_dim: u32,
        _top_k: u32,
        _batch_size: u32,
        _command_buffer: &mut VkCommandBufferEncoding,
    ) {
        unreachable!("unsupported MoE combine should not be constructed")
    }
}
