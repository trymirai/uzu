use crate::{
    backends::{
        common::{
            Backend, BufferRef, CommandBufferEncoding,
            kernel::{AttentionArguments, AttentionInput, AttentionKernel, AttentionKernelConfig},
        },
        vulkan::{
            AttentionGemmVulkanKernel, AttentionGroupedVulkanKernel, AttentionSinglePassVulkanKernel,
            AttentionTwoPass1VulkanKernel, AttentionTwoPass2VulkanKernel, Error, SigmoidGateVulkanKernel, VkBuffer,
            VkCommandBufferEncoding, VkContext, Vulkan,
            gpu_types::{
                attention_gemm::ATTENTION_GEMM_SHARED_BYTES,
                attention_grouped::{ATTENTION_GROUPED_KEYS, ATTENTION_GROUPED_ROWS},
                attention_two_pass::ATTENTION_TWO_PASS_BLOCKS,
            },
        },
    },
    data_type::DataType,
};

/// Two-pass kernels for at most TWO_PASS_MAX_SUFFIX queries, which split the keys into blocks whatever their count.
const TWO_PASS_MAX_SUFFIX: u32 = 8;
/// AttentionGrouped for 2 to GROUPED_MAX_SUFFIX queries over a key-value head's tile of rows and a block of keys or
/// more, split in whole blocks.
const GROUPED_MAX_SUFFIX: u32 = 64;

pub struct VkAttentionKernel {
    config: AttentionKernelConfig,
    /// Without and with a trie, for more than TWO_PASS_MAX_SUFFIX queries of head dimensions AttentionGemm lacks.
    single_pass: [AttentionSinglePassVulkanKernel; 2],
    /// Without and with a trie, for more than TWO_PASS_MAX_SUFFIX BF16 queries of 64, 128 or 256 dimensions on devices
    /// with its workgroup memory.
    gemm: Option<[AttentionGemmVulkanKernel; 2]>,
    gate: Option<SigmoidGateVulkanKernel>,
    /// Without and with a trie.
    two_pass: [AttentionTwoPass1VulkanKernel; 2],
    /// Without and with a gate.
    merge: [AttentionTwoPass2VulkanKernel; 2],
    /// Without and with a trie, on 16-bit tensor cores for 128 or 256 dimensions without sinks.
    grouped: Option<[AttentionGroupedVulkanKernel; 2]>,
}

impl AttentionKernel for VkAttentionKernel {
    type Backend = Vulkan;
    type Error = Error;

    fn new(
        context: &VkContext,
        config: AttentionKernelConfig,
    ) -> Result<Self, Error> {
        let arc = context.arc();
        let single_pass = |is_trie| {
            AttentionSinglePassVulkanKernel::create(
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
        let gemm = |is_trie| {
            AttentionGemmVulkanKernel::new(
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
        let grouped = |is_trie| {
            AttentionGroupedVulkanKernel::new(
                &arc,
                config.data_type,
                config.head_dim,
                config.is_kv_cache_ring,
                config.is_causal,
                is_trie,
                config.sliding_window_size.is_some(),
            )
        };
        let has_gemm = [64, 128, 256].contains(&config.head_dim)
            && config.data_type == DataType::BF16
            && context.physical_device().properties.limits.max_compute_shared_memory_size
                >= ATTENTION_GEMM_SHARED_BYTES;
        let has_grouped = [128, 256].contains(&config.head_dim)
            && !config.has_sinks
            && context.physical_device().capabilities.float_tensor_cores(config.data_type);
        Ok(Self {
            config,
            single_pass: [single_pass(false)?, single_pass(true)?],
            gemm: has_gemm.then(|| Ok::<_, Error>([gemm(false)?, gemm(true)?])).transpose()?,
            gate: config
                .gate_row_stride
                .map(|_| SigmoidGateVulkanKernel::create(&arc, config.data_type))
                .transpose()?,
            two_pass: [two_pass(false)?, two_pass(true)?],
            merge: [merge(false)?, merge(true)?],
            grouped: has_grouped.then(|| Ok::<_, Error>([grouped(false)?, grouped(true)?])).transpose()?,
        })
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
        let AttentionInput::FullPrecision(input) = input else {
            return Err(Error::NotSupported);
        };
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
        let grouped = self.grouped.as_ref().filter(|_| {
            (2..=GROUPED_MAX_SUFFIX).contains(&suffix_length)
                && gqa_factor * suffix_length >= ATTENTION_GROUPED_ROWS
                && sequence_length >= ATTENTION_GROUPED_KEYS
        });
        let output = command_buffer
            .allocate_scratch_for_shape(&[suffix_length, config.num_q_heads, config.head_dim], config.data_type)?;
        let (queries, keys, values) = (
            VkBuffer::resolve_ref(input.queries),
            VkBuffer::resolve_ref(input.keys),
            VkBuffer::resolve_ref(input.values),
        );
        let is_trie = usize::from(arguments.trie.is_some());
        let trie = arguments.trie.map(VkBuffer::resolve_ref);
        let scale = config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt());
        let row_stride = config.num_groups * config.head_dim;
        let gate = arguments.gate.zip(config.gate_row_stride);
        let command_buffer = &mut *command_buffer.debug_group("attention core");
        // SAFETY: the common contract makes callers pass queries [num_q_heads, suffix_length, head_dim], keys and values
        // of prefix + suffix rows [rows, num_groups, head_dim], suffix_length trie nodes, num_q_heads sinks and gate rows
        // of gate_row_stride; the partials, statistics and output are fresh scratch of their shapes.
        unsafe {
            if grouped.is_none() && suffix_length > TWO_PASS_MAX_SUFFIX {
                let output_ref = VkBuffer::resolve_ref(&output);
                match &self.gemm {
                    Some(gemm) => gemm[is_trie].encode(
                        queries,
                        keys,
                        values,
                        output_ref,
                        gqa_factor,
                        sequence_length,
                        config.head_dim,
                        row_stride,
                        config.head_dim,
                        row_stride,
                        arguments.cache.ring_params(),
                        scale,
                        trie,
                        config.sliding_window_size,
                        arguments.sinks.map(VkBuffer::resolve_ref),
                        config.num_q_heads,
                        suffix_length,
                        command_buffer,
                    ),
                    None => self.single_pass[is_trie].encode_ranges(
                        queries,
                        keys,
                        values,
                        output_ref,
                        gqa_factor,
                        sequence_length,
                        config.head_dim,
                        row_stride,
                        config.head_dim,
                        row_stride,
                        arguments.cache.ring_params(),
                        scale,
                        trie,
                        config.sliding_window_size,
                        arguments.sinks.map(VkBuffer::resolve_ref),
                        config.num_q_heads,
                        suffix_length,
                        command_buffer,
                    ),
                }
                if let (Some((gate, gate_row_stride)), Some(gate_kernel)) = (gate, &self.gate) {
                    gate_kernel.encode_ranges(
                        VkBuffer::resolve_ref(gate),
                        VkBuffer::resolve_ref(&output),
                        config.num_q_heads * config.head_dim,
                        suffix_length,
                        gate_row_stride,
                        command_buffer,
                    );
                }
                return Ok(output);
            }
            let blocks = match grouped {
                Some(_) => sequence_length.div_ceil(ATTENTION_GROUPED_KEYS).min(ATTENTION_TWO_PASS_BLOCKS),
                None => ATTENTION_TWO_PASS_BLOCKS,
            };
            let rows = config.num_q_heads * suffix_length * blocks;
            let partials = command_buffer.allocate_scratch_for_shape(&[rows, config.head_dim], DataType::F32)?;
            let sums = command_buffer.allocate_scratch_for_shape(&[rows], DataType::F32)?;
            let maxs = command_buffer.allocate_scratch_for_shape(&[rows], DataType::F32)?;
            match grouped {
                Some(grouped) => grouped[is_trie].encode(
                    queries,
                    keys,
                    values,
                    VkBuffer::resolve_ref(&partials),
                    VkBuffer::resolve_ref(&sums),
                    VkBuffer::resolve_ref(&maxs),
                    gqa_factor,
                    sequence_length,
                    config.head_dim,
                    row_stride,
                    config.head_dim,
                    row_stride,
                    arguments.cache.ring_params(),
                    scale,
                    config.num_q_heads,
                    suffix_length,
                    trie,
                    config.sliding_window_size,
                    blocks,
                    command_buffer,
                ),
                None => self.two_pass[is_trie].encode(
                    queries,
                    keys,
                    values,
                    VkBuffer::resolve_ref(&partials),
                    VkBuffer::resolve_ref(&sums),
                    VkBuffer::resolve_ref(&maxs),
                    gqa_factor,
                    sequence_length,
                    config.head_dim,
                    row_stride,
                    config.head_dim,
                    row_stride,
                    arguments.cache.ring_params(),
                    scale,
                    config.num_q_heads,
                    suffix_length,
                    trie,
                    config.sliding_window_size,
                    arguments.sinks.map(VkBuffer::resolve_ref),
                    command_buffer,
                ),
            }
            self.merge[usize::from(gate.is_some())].encode(
                VkBuffer::resolve_ref(&partials),
                VkBuffer::resolve_ref(&sums),
                VkBuffer::resolve_ref(&maxs),
                VkBuffer::resolve_ref(&output),
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
