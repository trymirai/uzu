use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, Kernels,
            kernel::{
                DeltaNetPrefillPrepKernel,
                delta_net_chunked_prefill::{CHUNK_SIZE, DeltaNetChunkedPrefill, DeltaNetChunkedPrefillArgs},
            },
        },
        vulkan::{
            DeltaNetChunkedOutputAndStateVulkanKernel, DeltaNetChunkedPrecomputeVulkanKernel, Error, VkBuffer,
            VkCommandBufferEncoding, VkContext, Vulkan, VulkanKernels,
            gpu_types::delta_net::{DELTA_NET_CHUNKED_HEAD_K_DIM, DELTA_NET_CHUNKED_MAX_GROUPS},
        },
    },
    data_type::DataType,
};

/// The shortest suffix worth the chunked kernels' extra dispatches, as Metal's.
const MIN_SUFFIX_LEN: u32 = 256;
const INNER_DATA_TYPE: DataType = DataType::F32;

/// The chunked gated delta rule in two stages after DeltaNetPrefillPrep.
pub struct VkDeltaNetChunkedPrefill {
    prep: <VulkanKernels as Kernels>::DeltaNetPrefillPrepKernel,
    precompute: DeltaNetChunkedPrecomputeVulkanKernel,
    output_and_state: DeltaNetChunkedOutputAndStateVulkanKernel,
}

impl DeltaNetChunkedPrefill for VkDeltaNetChunkedPrefill {
    type Backend = Vulkan;

    fn new(
        context: &VkContext,
        outer_data_type: DataType,
        head_dim: u32,
        groups_per_head: u32,
    ) -> Result<Option<Self>, Error> {
        if !matches!(outer_data_type, DataType::F32 | DataType::BF16)
            || head_dim != DELTA_NET_CHUNKED_HEAD_K_DIM
            || groups_per_head > DELTA_NET_CHUNKED_MAX_GROUPS
        {
            return Ok(None);
        }
        let arc = context.arc();
        Ok(Some(Self {
            prep: DeltaNetPrefillPrepKernel::new(context, outer_data_type, INNER_DATA_TYPE, head_dim, true, false)?,
            precompute: DeltaNetChunkedPrecomputeVulkanKernel::new(&arc)?,
            output_and_state: DeltaNetChunkedOutputAndStateVulkanKernel::new(&arc, outer_data_type)?,
        }))
    }

    fn should_use(
        &self,
        suffix_len: u32,
    ) -> bool {
        suffix_len >= MIN_SUFFIX_LEN
    }

    fn encode(
        &self,
        args: DeltaNetChunkedPrefillArgs<'_, Vulkan>,
        command_buffer: &mut VkCommandBufferEncoding,
    ) -> Result<(), Error> {
        let suffix_len = args.suffix_len;
        let (num_heads, num_groups) = (args.num_heads, args.num_groups);
        let num_chunks = suffix_len.div_ceil(CHUNK_SIZE);

        let mut q_norm = command_buffer.allocate_scratch_for_shape(&[suffix_len, args.key_dim], INNER_DATA_TYPE)?;
        let mut k_norm = command_buffer.allocate_scratch_for_shape(&[suffix_len, args.key_dim], INNER_DATA_TYPE)?;
        let mut beta = command_buffer.allocate_scratch_for_shape(&[suffix_len, num_heads], INNER_DATA_TYPE)?;
        let mut log_decay = command_buffer.allocate_scratch_for_shape(&[suffix_len, num_heads], INNER_DATA_TYPE)?;
        let g = command_buffer.allocate_scratch_for_shape(&[num_heads, suffix_len], INNER_DATA_TYPE)?;
        let qk = command_buffer
            .allocate_scratch_for_shape(&[num_chunks, num_heads, CHUNK_SIZE, CHUNK_SIZE], INNER_DATA_TYPE)?;
        let t_mat = command_buffer
            .allocate_scratch_for_shape(&[num_chunks, num_heads, CHUNK_SIZE, CHUNK_SIZE], INNER_DATA_TYPE)?;

        DeltaNetPrefillPrepKernel::encode(
            &self.prep,
            args.in_projected,
            args.a_log,
            args.dt_bias,
            &mut q_norm,
            &mut k_norm,
            None::<&mut <Vulkan as Backend>::ScratchBuffer>,
            &mut beta,
            &mut log_decay,
            num_heads,
            num_groups,
            args.key_dim,
            args.value_dim,
            suffix_len,
            command_buffer,
        );
        // SAFETY (every stage): the scratch ranges above have the shapes of these sizes, `in_projected` holds
        // [suffix_len, 2 key_dim + 2 value_dim + 2 num_heads] rows, `ssm_state` [num_heads, value_head_dim, 128] and
        // `delta_output` [suffix_len, value_dim]; each stage writes only ranges no other argument of it aliases, and a
        // stage reads only entries an earlier stage wrote.
        unsafe {
            self.precompute.encode(
                VkBuffer::resolve_ref(&q_norm),
                VkBuffer::resolve_ref(&k_norm),
                VkBuffer::resolve_ref(&log_decay),
                VkBuffer::resolve_ref(&beta),
                VkBuffer::resolve_ref(&g),
                VkBuffer::resolve_ref(&qk),
                VkBuffer::resolve_ref(&t_mat),
                num_heads,
                num_groups,
                args.key_dim,
                suffix_len,
                command_buffer,
            );
            self.output_and_state.encode(
                VkBuffer::resolve_ref(&q_norm),
                VkBuffer::resolve_ref(&k_norm),
                VkBuffer::resolve_ref(args.in_projected),
                VkBuffer::resolve_ref(&qk),
                VkBuffer::resolve_ref(&t_mat),
                VkBuffer::resolve_ref(&g),
                VkBuffer::resolve_ref(&beta),
                VkBuffer::resolve_ref(&*args.ssm_state),
                VkBuffer::resolve_ref(&*args.delta_output),
                num_heads,
                num_groups,
                args.value_head_dim,
                args.key_dim,
                args.value_dim,
                suffix_len,
                command_buffer,
            );
        }
        Ok(())
    }
}
