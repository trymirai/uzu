use super::{
    Error, VkAttentionKernel, VkCommandBufferEncoding, VkContext, VkDeltaNetChunkedPrefill, VkDeltaNetTreeVerify,
    VkDeltaNetUpdateKernel, VkMatmulKernel, VkRadixTopKSmall, Vulkan,
};
use crate::{
    backends::common::{
        BufferMut, BufferRef, Kernels,
        gpu_types::kv_cache_update,
        kernel::{AttentionPrepareInt8Kernel, KVCacheUpdateInt8Kernel, Unsupported},
    },
    data_type::DataType,
};

pub struct VulkanKernels;

impl Kernels for VulkanKernels {
    type Backend = Vulkan;

    autogen_kernels!();
    type AttentionPrepareInt8Kernel = Unsupported<Vulkan>;
    type KVCacheUpdateInt8Kernel = Unsupported<Vulkan>;
    type AttentionKernel = VkAttentionKernel;
    type DeltaNetChunkedPrefill = VkDeltaNetChunkedPrefill;
    type DeltaNetTreeVerify = VkDeltaNetTreeVerify;
    type DeltaNetUpdateKernel = VkDeltaNetUpdateKernel;
    type MatmulKernel = VkMatmulKernel;
    type RadixTopKSmall = VkRadixTopKSmall;
}

impl AttentionPrepareInt8Kernel for Unsupported<Vulkan> {
    type Backend = Vulkan;

    fn new(
        _context: &VkContext,
        _element_type: DataType,
        _rope_type: DataType,
        _head_dim: u32,
        _has_rope: bool,
        _quantize_queries: bool,
        _num_kv_heads: u32,
        _append_kv: bool,
    ) -> Result<Self, Error> {
        Err(Error::NotSupported)
    }

    fn encode(
        &self,
        _qkvg: impl BufferRef<Backend = Vulkan>,
        _queries: Option<impl BufferMut<Backend = Vulkan>>,
        _q_codes: Option<impl BufferMut<Backend = Vulkan>>,
        _q_scales: Option<impl BufferMut<Backend = Vulkan>>,
        _k_codes: Option<impl BufferMut<Backend = Vulkan>>,
        _k_scales: Option<impl BufferMut<Backend = Vulkan>>,
        _v_codes: Option<impl BufferMut<Backend = Vulkan>>,
        _v_scales: Option<impl BufferMut<Backend = Vulkan>>,
        _cosines: Option<impl BufferRef<Backend = Vulkan>>,
        _sines: Option<impl BufferRef<Backend = Vulkan>>,
        _num_q_heads: u32,
        _rope_dim: Option<u32>,
        _kv_token_offset: u32,
        _input_row_stride: u32,
        _suffix_length: u32,
        _query_rows: u32,
        _command_buffer: &mut VkCommandBufferEncoding,
    ) {
        unreachable!("unsupported INT8 attention prepare should not be constructed")
    }
}

impl KVCacheUpdateInt8Kernel for Unsupported<Vulkan> {
    type Backend = Vulkan;

    fn new(
        _context: &VkContext,
        _head_dim: u32,
        _num_kv_heads: u32,
    ) -> Result<Self, Error> {
        Err(Error::NotSupported)
    }

    fn encode(
        &self,
        _in_place_k_codes: impl BufferMut<Backend = Vulkan>,
        _in_place_v_codes: impl BufferMut<Backend = Vulkan>,
        _in_place_k_scales: impl BufferMut<Backend = Vulkan>,
        _in_place_v_scales: impl BufferMut<Backend = Vulkan>,
        _copies: &[kv_cache_update::Copy],
        _copy_count: u32,
        _command_buffer: &mut VkCommandBufferEncoding,
    ) {
        unreachable!("unsupported INT8 KV cache update should not be constructed")
    }
}
