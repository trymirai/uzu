use crate::{
    backends::{
        common::{
            Backend, BufferRef, CommandBufferEncoding,
            kernel::{AttentionArguments, AttentionInput, AttentionKernel, AttentionKernelConfig, kv_layout::KVLayout},
        },
        vulkan::{
            AttentionGemmVulkanKernel, AttentionSinglePassVulkanKernel, AttentionTwoPass1VulkanKernel,
            AttentionTwoPass2VulkanKernel, Error, VkBuffer, VkCommandBufferEncoding, VkContext, Vulkan,
            gpu_types::{
                attention_gemm::{ATTENTION_GEMM_BLOCK_KEYS, ATTENTION_GEMM_SHARED_BYTES},
                attention_two_pass::ATTENTION_TWO_PASS_BLOCKS,
            },
        },
    },
    data_type::DataType,
};

const TWO_PASS_MAX_SUFFIX: u32 = 8;
const GEMM_SPLIT_MAX_SUFFIX: u32 = 64;

enum VkAttentionManyQueries {
    Gemm([AttentionGemmVulkanKernel; 2]),
    SinglePass([AttentionSinglePassVulkanKernel; 2]),
}

pub struct VkAttentionKernel {
    config: AttentionKernelConfig,
    many_queries: VkAttentionManyQueries,
    two_pass: [AttentionTwoPass1VulkanKernel; 2],
    merge: AttentionTwoPass2VulkanKernel,
}

impl AttentionKernel for VkAttentionKernel {
    type Backend = Vulkan;
    type Error = Error;

    fn new(
        context: &VkContext,
        config: AttentionKernelConfig,
    ) -> Result<Self, Error> {
        let arc = context.arc();
        let is_sliding_window = config.sliding_window_size.is_some();
        let single_pass = |is_trie| {
            AttentionSinglePassVulkanKernel::new(
                &arc,
                config.head_dim,
                config.is_kv_cache_ring,
                config.is_causal,
                is_trie,
                is_sliding_window,
            )
        };
        let gemm = |is_trie| {
            AttentionGemmVulkanKernel::new(
                &arc,
                config.head_dim,
                config.is_kv_cache_ring,
                config.is_causal,
                is_trie,
                is_sliding_window,
            )
        };
        let two_pass = |is_trie| {
            AttentionTwoPass1VulkanKernel::new(
                &arc,
                config.head_dim,
                config.is_kv_cache_ring,
                config.is_causal,
                is_trie,
                is_sliding_window,
            )
        };
        let has_gemm = [64, 128, 256].contains(&config.head_dim)
            && context.physical_device().limits.max_shared_memory_size >= ATTENTION_GEMM_SHARED_BYTES;
        Ok(Self {
            config,
            many_queries: if has_gemm {
                VkAttentionManyQueries::Gemm([gemm(false)?, gemm(true)?])
            } else {
                VkAttentionManyQueries::SinglePass([single_pass(false)?, single_pass(true)?])
            },
            two_pass: [two_pass(false)?, two_pass(true)?],
            merge: AttentionTwoPass2VulkanKernel::new(
                &arc,
                config.data_type,
                config.head_dim,
                config.has_sinks,
                config.gate_row_stride.is_some(),
            )?,
        })
    }

    fn kv_layout(&self) -> KVLayout {
        KVLayout::new(self.config.head_dim, self.config.num_groups, 2, 0)
    }

    fn encode(
        &self,
        input: AttentionInput<
            impl BufferRef<Backend = Vulkan>,
            impl BufferRef<Backend = Vulkan>,
            impl BufferRef<Backend = Vulkan>,
        >,
        arguments: AttentionArguments<'_, Vulkan, impl BufferRef<Backend = Vulkan>, impl BufferRef<Backend = Vulkan>>,
        command_buffer: &mut VkCommandBufferEncoding,
    ) -> Result<<Vulkan as Backend>::ScratchBuffer, Error> {
        let config = self.config;
        let suffix_length = arguments.suffix_length;
        let sequence_length = arguments.cache.prefix_len() + suffix_length;
        command_buffer.declare_work(
            4 * u64::from(config.num_q_heads)
                * u64::from(suffix_length)
                * u64::from(sequence_length)
                * u64::from(config.head_dim),
        );
        let gqa_factor = config.num_q_heads / config.num_groups;
        let output = command_buffer
            .allocate_scratch_for_shape(&[suffix_length, config.num_q_heads, config.head_dim], config.data_type)?;
        let blocks = if suffix_length <= TWO_PASS_MAX_SUFFIX {
            ATTENTION_TWO_PASS_BLOCKS
        } else if matches!(self.many_queries, VkAttentionManyQueries::Gemm(_)) && suffix_length <= GEMM_SPLIT_MAX_SUFFIX
        {
            sequence_length.div_ceil(ATTENTION_GEMM_BLOCK_KEYS).min(ATTENTION_TWO_PASS_BLOCKS)
        } else {
            1
        };
        let rows = config.num_q_heads * suffix_length * blocks;
        let partials = command_buffer.allocate_scratch_for_shape(&[rows, config.head_dim], DataType::F32)?;
        let sums = command_buffer.allocate_scratch_for_shape(&[rows], DataType::F32)?;
        let maxs = command_buffer.allocate_scratch_for_shape(&[rows], DataType::F32)?;
        let q_codes = VkBuffer::resolve_ref(input.query_codes);
        let q_scales = VkBuffer::resolve_ref(input.query_scales);
        let k_codes = VkBuffer::resolve_ref(input.k_codes);
        let k_scales = VkBuffer::resolve_ref(input.k_scales);
        let v_codes = VkBuffer::resolve_ref(input.v_codes);
        let v_scales = VkBuffer::resolve_ref(input.v_scales);
        let partials_ref = VkBuffer::resolve_ref(&partials);
        let sums_ref = VkBuffer::resolve_ref(&sums);
        let maxs_ref = VkBuffer::resolve_ref(&maxs);
        let is_trie = usize::from(arguments.trie.is_some());
        let trie = arguments.trie.map(VkBuffer::resolve_ref);
        let ring_params = arguments.cache.ring_params();
        let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());
        let command_buffer = &mut *command_buffer.debug_group("attention core");
        macro_rules! partials {
            ($kernel:expr $(, $blocks:expr)?) => {
                $kernel.encode(
                    q_codes,
                    q_scales,
                    k_codes,
                    k_scales,
                    v_codes,
                    v_scales,
                    partials_ref.clone(),
                    sums_ref.clone(),
                    maxs_ref.clone(),
                    gqa_factor,
                    config.num_groups,
                    sequence_length,
                    ring_params,
                    scale,
                    trie,
                    config.sliding_window_size,
                    config.num_q_heads,
                    suffix_length,
                    $($blocks,)?
                    command_buffer,
                )
            };
        }
        unsafe {
            match &self.many_queries {
                _ if suffix_length <= TWO_PASS_MAX_SUFFIX => partials!(self.two_pass[is_trie]),
                VkAttentionManyQueries::Gemm(gemm) => partials!(gemm[is_trie], blocks),
                VkAttentionManyQueries::SinglePass(single_pass) => partials!(single_pass[is_trie]),
            }
            self.merge.encode(
                partials_ref,
                sums_ref,
                maxs_ref,
                VkBuffer::resolve_ref(&output),
                arguments.sinks.map(VkBuffer::resolve_ref),
                arguments.gate.map(VkBuffer::resolve_ref),
                config.num_q_heads,
                suffix_length,
                config.gate_row_stride,
                blocks,
                command_buffer,
            );
        }
        Ok(output)
    }
}
