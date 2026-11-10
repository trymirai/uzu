use crate::{
    backends::{
        common::{
            BufferMut, BufferRef, Kernels,
            gpu_types::confidence_head::ConfidenceHeadParams,
            kernel::{
                ConfidenceHeadEdgesKernel, ConfidenceHeadNodesKernel, ConfidenceHeadPoolsKernel,
                ConfidenceHeadTreeAttentionKernel, ConfidenceHeadTreeInputKernel, ConfidenceHeadTreeRelationsKernel,
                MoeCombineKernel, MoeGemvKernel, MoeRouterTopKKernel, Unsupported,
            },
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
    // No released drafter ships a confidence head yet, so Vulkan has none of its kernels.
    type ConfidenceHeadEdgesKernel = Unsupported<Vulkan>;
    type ConfidenceHeadNodesKernel = Unsupported<Vulkan>;
    type ConfidenceHeadPoolsKernel = Unsupported<Vulkan>;
    type ConfidenceHeadTreeAttentionKernel = Unsupported<Vulkan>;
    type ConfidenceHeadTreeInputKernel = Unsupported<Vulkan>;
    type ConfidenceHeadTreeRelationsKernel = Unsupported<Vulkan>;
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

impl ConfidenceHeadEdgesKernel for Unsupported<Vulkan> {
    type Backend = Vulkan;

    fn new(
        _context: &VkContext,
        _model_dim: u32,
        _edge_hidden: u32,
        _refine: bool,
    ) -> Result<Self, Error> {
        Err(Error::NotSupported)
    }

    fn encode(
        &self,
        _node_metadata: Option<impl BufferRef<Backend = Vulkan>>,
        _node_valid: Option<impl BufferRef<Backend = Vulkan>>,
        _candidate_ids: Option<impl BufferRef<Backend = Vulkan>>,
        _pool_features: Option<impl BufferRef<Backend = Vulkan>>,
        _slot_stats: Option<impl BufferRef<Backend = Vulkan>>,
        _slot_rows: impl BufferMut<Backend = Vulkan>,
        _row_slots: Option<impl BufferMut<Backend = Vulkan>>,
        _nodes: impl BufferRef<Backend = Vulkan>,
        _node_shifts: Option<impl BufferRef<Backend = Vulkan>>,
        _child_shifts: Option<impl BufferRef<Backend = Vulkan>>,
        _token_features: impl BufferRef<Backend = Vulkan>,
        _child_numbers: impl BufferMut<Backend = Vulkan>,
        _child_pools: impl BufferMut<Backend = Vulkan>,
        _scores: impl BufferRef<Backend = Vulkan>,
        _frontier: impl BufferMut<Backend = Vulkan>,
        _packed_tree: impl BufferMut<Backend = Vulkan>,
        _child_inputs: Option<impl BufferMut<Backend = Vulkan>>,
        _input_norm: impl BufferRef<Backend = Vulkan>,
        _entry_weights: impl BufferRef<Backend = Vulkan>,
        _entry_biases: impl BufferRef<Backend = Vulkan>,
        _block_norm: impl BufferRef<Backend = Vulkan>,
        _up_weights: impl BufferRef<Backend = Vulkan>,
        _up_biases: impl BufferRef<Backend = Vulkan>,
        _down_weights: impl BufferRef<Backend = Vulkan>,
        _down_biases: impl BufferRef<Backend = Vulkan>,
        _final_norm: impl BufferRef<Backend = Vulkan>,
        _readout_weights: impl BufferRef<Backend = Vulkan>,
        _readout_biases: impl BufferRef<Backend = Vulkan>,
        _params: ConfidenceHeadParams,
        _groups: u32,
        _command_buffer: &mut VkCommandBufferEncoding,
    ) {
        unreachable!("unsupported confidence-head edges should not be constructed")
    }
}

impl ConfidenceHeadNodesKernel for Unsupported<Vulkan> {
    type Backend = Vulkan;

    fn new(
        _context: &VkContext,
        _model_dim: u32,
        _pool_hidden: u32,
    ) -> Result<Self, Error> {
        Err(Error::NotSupported)
    }

    fn encode(
        &self,
        _residual_logits: impl BufferRef<Backend = Vulkan>,
        _candidate_logits: impl BufferRef<Backend = Vulkan>,
        _pool_features: impl BufferRef<Backend = Vulkan>,
        _round_metadata: impl BufferRef<Backend = Vulkan>,
        _round_valid: impl BufferRef<Backend = Vulkan>,
        _draft: impl BufferRef<Backend = Vulkan>,
        _pool_entry_weights: impl BufferRef<Backend = Vulkan>,
        _pool_entry_biases: impl BufferRef<Backend = Vulkan>,
        _pool_hidden_weights: impl BufferRef<Backend = Vulkan>,
        _pool_hidden_biases: impl BufferRef<Backend = Vulkan>,
        _pool_out_weights: impl BufferRef<Backend = Vulkan>,
        _pool_out_biases: impl BufferRef<Backend = Vulkan>,
        _scores: impl BufferMut<Backend = Vulkan>,
        _nodes: impl BufferMut<Backend = Vulkan>,
        _node_inputs: impl BufferMut<Backend = Vulkan>,
        _params: ConfidenceHeadParams,
        _rows: u32,
        _command_buffer: &mut VkCommandBufferEncoding,
    ) {
        unreachable!("unsupported confidence-head nodes should not be constructed")
    }
}

impl ConfidenceHeadPoolsKernel for Unsupported<Vulkan> {
    type Backend = Vulkan;

    fn new(_context: &VkContext) -> Result<Self, Error> {
        Err(Error::NotSupported)
    }

    fn encode(
        &self,
        _candidate_logits: impl BufferRef<Backend = Vulkan>,
        _candidate_ids: impl BufferRef<Backend = Vulkan>,
        _depth_seeds: impl BufferRef<Backend = Vulkan>,
        _pool_features: impl BufferMut<Backend = Vulkan>,
        _params: ConfidenceHeadParams,
        _depths: u32,
        _command_buffer: &mut VkCommandBufferEncoding,
    ) {
        unreachable!("unsupported confidence-head pools should not be constructed")
    }
}

impl ConfidenceHeadTreeAttentionKernel for Unsupported<Vulkan> {
    type Backend = Vulkan;

    fn new(
        _context: &VkContext,
        _num_heads: u32,
        _head_dim: u32,
    ) -> Result<Self, Error> {
        Err(Error::NotSupported)
    }

    fn encode(
        &self,
        _qkv: impl BufferRef<Backend = Vulkan>,
        _relations: impl BufferRef<Backend = Vulkan>,
        _relation_bias: impl BufferRef<Backend = Vulkan>,
        _output: impl BufferMut<Backend = Vulkan>,
        _rows: u32,
        _command_buffer: &mut VkCommandBufferEncoding,
    ) {
        unreachable!("unsupported confidence-head tree attention should not be constructed")
    }
}

impl ConfidenceHeadTreeInputKernel for Unsupported<Vulkan> {
    type Backend = Vulkan;

    fn new(_context: &VkContext) -> Result<Self, Error> {
        Err(Error::NotSupported)
    }

    fn encode(
        &self,
        _depth_embedding: impl BufferRef<Backend = Vulkan>,
        _child_in: impl BufferRef<Backend = Vulkan>,
        _node_in: impl BufferRef<Backend = Vulkan>,
        _frontier: impl BufferRef<Backend = Vulkan>,
        _row_slots: impl BufferRef<Backend = Vulkan>,
        _output: impl BufferMut<Backend = Vulkan>,
        _rows: u32,
        _width: u32,
        _command_buffer: &mut VkCommandBufferEncoding,
    ) {
        unreachable!("unsupported confidence-head tree input should not be constructed")
    }
}

impl ConfidenceHeadTreeRelationsKernel for Unsupported<Vulkan> {
    type Backend = Vulkan;

    fn new(_context: &VkContext) -> Result<Self, Error> {
        Err(Error::NotSupported)
    }

    fn encode(
        &self,
        _frontier: impl BufferRef<Backend = Vulkan>,
        _slot_ancestors: impl BufferRef<Backend = Vulkan>,
        _slot_rows: impl BufferRef<Backend = Vulkan>,
        _relations: impl BufferMut<Backend = Vulkan>,
        _rows: u32,
        _ancestor_stride: u32,
        _command_buffer: &mut VkCommandBufferEncoding,
    ) {
        unreachable!("unsupported confidence-head tree relations should not be constructed")
    }
}
