use super::{
    AttentionTwoPass1VulkanKernel, AttentionTwoPass2VulkanKernel, Error, VkBuffer, VkCommandBufferEncoding, VkContext,
    Vulkan,
};
use crate::{
    backends::common::{
        Backend, BufferRef, CommandBufferEncoding,
        kernel::{AttentionArguments, AttentionKernel, AttentionKernelConfig, attention::SinglePassAttention},
    },
    data_type::DataType,
};

/// Metal's non-MXU routing over the kernels Vulkan has: more than SINGLE_PASS_KV_THRESHOLD keys take the two-pass
/// kernels, everything else AttentionSinglePass. Metal sends more than 8 queries to AttentionGemm on this device class,
/// which Vulkan lacks, so those stay single pass rather than allocating per-query partials for whole prompts.
const SINGLE_PASS_KV_THRESHOLD: u32 = 1_024;
const TWO_PASS_MAX_SUFFIX: u32 = 8;
const TWO_PASS_BLOCKS: u32 = 32;

pub struct VkAttentionKernel {
    config: AttentionKernelConfig,
    single_pass: SinglePassAttention<Vulkan>,
    /// Without and with a trie.
    two_pass: [AttentionTwoPass1VulkanKernel; 2],
    /// Without and with a gate.
    merge: [AttentionTwoPass2VulkanKernel; 2],
}

impl AttentionKernel for VkAttentionKernel {
    type Backend = Vulkan;

    fn new(
        context: &VkContext,
        config: AttentionKernelConfig,
    ) -> Result<Self, Error> {
        let arc = context.arc();
        let two_pass = |is_trie| {
            AttentionTwoPass1VulkanKernel::new(
                &arc,
                config.data_type,
                config.head_dim,
                config.has_sinks,
                config.is_kv_cache_ring,
                config.is_causal,
                is_trie,
                config.sliding_window_size.is_some(),
            )
        };
        let merge = |has_gate| AttentionTwoPass2VulkanKernel::new(&arc, config.data_type, config.head_dim, has_gate);
        Ok(Self {
            config,
            single_pass: SinglePassAttention::create(context, config)?,
            two_pass: [two_pass(false)?, two_pass(true)?],
            merge: [merge(false)?, merge(true)?],
        })
    }

    fn encode(
        &self,
        arguments: AttentionArguments<
            '_,
            Vulkan,
            impl BufferRef<Backend = Vulkan>,
            impl BufferRef<Backend = Vulkan>,
            impl BufferRef<Backend = Vulkan>,
            impl BufferRef<Backend = Vulkan>,
            impl BufferRef<Backend = Vulkan>,
        >,
        command_buffer: &mut VkCommandBufferEncoding,
    ) -> Result<<Vulkan as Backend>::ScratchBuffer, Error> {
        let config = self.config;
        let suffix_length = arguments.suffix_length;
        let sequence_length = arguments.cache.prefix_len() + suffix_length;
        if suffix_length > TWO_PASS_MAX_SUFFIX || sequence_length <= SINGLE_PASS_KV_THRESHOLD {
            return self.single_pass.encode_gated(arguments, command_buffer);
        }
        let output = command_buffer
            .allocate_scratch_for_shape(&[suffix_length, config.num_q_heads, config.head_dim], config.data_type)?;
        let rows = config.num_q_heads * suffix_length * TWO_PASS_BLOCKS;
        let partials = command_buffer.allocate_scratch_for_shape(&[rows, config.head_dim], DataType::F32)?;
        let sums = command_buffer.allocate_scratch_for_shape(&[rows], DataType::F32)?;
        let maxs = command_buffer.allocate_scratch_for_shape(&[rows], DataType::F32)?;
        let gate_row_stride = arguments.gate.and(config.gate_row_stride);
        {
            let command_buffer = &mut *command_buffer.debug_group("attention core");
            // SAFETY: the common contract makes callers pass queries [num_q_heads, suffix_length, head_dim], keys and values
            // of prefix + suffix rows [rows, num_groups, head_dim], suffix_length trie nodes, num_q_heads sinks and gate rows
            // of gate_row_stride; the partials, statistics and output are fresh scratch of their shapes.
            unsafe {
                self.two_pass[usize::from(arguments.trie.is_some())].encode(
                    VkBuffer::resolve_ref(arguments.queries),
                    VkBuffer::resolve_ref(arguments.keys),
                    VkBuffer::resolve_ref(arguments.values),
                    VkBuffer::resolve_ref(&partials),
                    VkBuffer::resolve_ref(&sums),
                    VkBuffer::resolve_ref(&maxs),
                    config.num_q_heads / config.num_groups,
                    sequence_length,
                    config.head_dim,
                    config.num_groups * config.head_dim,
                    config.head_dim,
                    config.num_groups * config.head_dim,
                    arguments.cache.ring_params(),
                    config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt()),
                    config.num_q_heads,
                    suffix_length,
                    arguments.trie.map(VkBuffer::resolve_ref),
                    config.sliding_window_size,
                    arguments.sinks.map(VkBuffer::resolve_ref),
                    command_buffer,
                );
                self.merge[usize::from(gate_row_stride.is_some())].encode(
                    VkBuffer::resolve_ref(&partials),
                    VkBuffer::resolve_ref(&sums),
                    VkBuffer::resolve_ref(&maxs),
                    VkBuffer::resolve_ref(&output),
                    gate_row_stride.and(arguments.gate.map(VkBuffer::resolve_ref)),
                    config.num_q_heads,
                    suffix_length,
                    gate_row_stride,
                    command_buffer,
                );
            }
        }
        Ok(output)
    }
}
