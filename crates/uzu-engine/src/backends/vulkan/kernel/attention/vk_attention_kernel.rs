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

/// Two-pass kernels for at most TWO_PASS_MAX_SUFFIX queries, which split the keys into blocks whatever their count.
const TWO_PASS_MAX_SUFFIX: u32 = 8;
/// AttentionGemm splits the keys of at most GEMM_SPLIT_MAX_SUFFIX queries, as Metal's attention does.
const GEMM_SPLIT_MAX_SUFFIX: u32 = 64;

/// Attention over AttentionPrepare's INT8 codes in rows [key][kv head], scales [key][kv head]: AttentionTwoPass1,
/// AttentionGemm or AttentionSinglePass write FP32 partials, which AttentionTwoPass2 merges with the sinks, rotates back
/// and gates.
pub struct VkAttentionKernel {
    config: AttentionKernelConfig,
    /// Without and with a trie, for more than TWO_PASS_MAX_SUFFIX queries without AttentionGemm.
    single_pass: Option<[AttentionSinglePassVulkanKernel; 2]>,
    /// Without and with a trie, for more than TWO_PASS_MAX_SUFFIX queries of 64, 128 or 256 dimensions on devices with
    /// its workgroup memory.
    gemm: Option<[AttentionGemmVulkanKernel; 2]>,
    /// Without and with a trie.
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
            && context.physical_device().properties.limits.max_compute_shared_memory_size
                >= ATTENTION_GEMM_SHARED_BYTES;
        Ok(Self {
            config,
            single_pass: (!has_gemm).then(|| Ok::<_, Error>([single_pass(false)?, single_pass(true)?])).transpose()?,
            gemm: has_gemm.then(|| Ok::<_, Error>([gemm(false)?, gemm(true)?])).transpose()?,
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

    /// Codes and scales in key order, keys allocated in pairs so the BF16 scale planes are whole words, as Vulkan fills
    /// take.
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
        // Scores and the weighted sum of values, each a multiply-add per query, key and head dimension.
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
        } else if self.gemm.is_some() && suffix_length <= GEMM_SPLIT_MAX_SUFFIX {
            // AttentionTwoPass1's split count, but no more splits than blocks of keys.
            sequence_length.div_ceil(ATTENTION_GEMM_BLOCK_KEYS).min(ATTENTION_TWO_PASS_BLOCKS)
        } else {
            1
        };
        let rows = config.num_q_heads * suffix_length * blocks;
        let partials = command_buffer.allocate_scratch_for_shape(&[rows, config.head_dim], DataType::F32)?;
        let sums = command_buffer.allocate_scratch_for_shape(&[rows], DataType::F32)?;
        let maxs = command_buffer.allocate_scratch_for_shape(&[rows], DataType::F32)?;
        let (q_codes, q_scales, k_codes, k_scales, v_codes, v_scales) = (
            VkBuffer::resolve_ref(input.query_codes),
            VkBuffer::resolve_ref(input.query_scales),
            VkBuffer::resolve_ref(input.k_codes),
            VkBuffer::resolve_ref(input.k_scales),
            VkBuffer::resolve_ref(input.v_codes),
            VkBuffer::resolve_ref(input.v_scales),
        );
        let (partials_ref, sums_ref, maxs_ref) =
            (VkBuffer::resolve_ref(&partials), VkBuffer::resolve_ref(&sums), VkBuffer::resolve_ref(&maxs));
        let is_trie = usize::from(arguments.trie.is_some());
        let trie = arguments.trie.map(VkBuffer::resolve_ref);
        let ring_params = arguments.cache.ring_params();
        let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());
        let gate = arguments.gate.zip(config.gate_row_stride);
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
        // SAFETY: the common contract makes callers pass query codes [num_q_heads, suffix_length, head_dim] with a scale
        // per row, key and value codes of prefix + suffix rows [rows, num_groups, head_dim] with a scale per row and
        // group, suffix_length trie nodes, num_q_heads sinks and gate rows of gate_row_stride; the partials, statistics
        // and output are fresh scratch of their shapes.
        unsafe {
            match (&self.gemm, &self.single_pass) {
                _ if suffix_length <= TWO_PASS_MAX_SUFFIX => partials!(self.two_pass[is_trie]),
                (Some(gemm), _) => partials!(gemm[is_trie], blocks),
                (None, Some(single_pass)) => partials!(single_pass[is_trie]),
                (None, None) => unreachable!("new builds the single-pass kernels without AttentionGemm"),
            }
            self.merge.encode(
                partials_ref,
                sums_ref,
                maxs_ref,
                VkBuffer::resolve_ref(&output),
                arguments.sinks.map(VkBuffer::resolve_ref),
                gate.map(|(gate, _)| VkBuffer::resolve_ref(gate)),
                config.num_q_heads,
                suffix_length,
                gate.map(|(_, gate_row_stride)| gate_row_stride),
                blocks,
                command_buffer,
            );
        }
        Ok(output)
    }
}
