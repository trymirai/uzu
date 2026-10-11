use super::{
    DeltaNetChunkedADiagInvVulkanKernel, DeltaNetChunkedCausalInvVulkanKernel, DeltaNetChunkedCumsumVulkanKernel,
    DeltaNetChunkedGramVulkanKernel, DeltaNetChunkedOutputAndStateVulkanKernel, Error, VkBuffer,
    VkCommandBufferEncoding, VkContext, Vulkan, VulkanKernels,
};
use crate::{
    backends::common::{
        Backend, CommandBufferEncoding, Kernels,
        kernel::{
            DeltaNetPrefillPrepKernel,
            delta_net_chunked_prefill::{DeltaNetChunkedPrefill, DeltaNetChunkedPrefillArgs},
        },
    },
    data_type::DataType,
};

/// Starting threshold inherited from Metal; Vulkan timing is measured separately.
const MIN_SUFFIX_LEN: u32 = 256;
const HEAD_K_DIM: u32 = 128;
/// Every stage's chunk, fixed at OutputAndState's 64 tokens.
const CHUNK_SIZE: u32 = 64;
const BLOCK_SIZE: u32 = 16;
const VT: u32 = 32;
const INNER_DATA_TYPE: DataType = DataType::F32;

/// Metal's five-stage chunked gated delta rule after the public DeltaNetPrefillPrep: cumulative log decay, the K Kᵀ and
/// decayed Q Kᵀ Grams, the packed strictly lower A with its diagonal-block inverses, the dense FP32 causal inverse T,
/// then output and final state, without MXU.
pub struct VkDeltaNetChunkedPrefill {
    prep: <VulkanKernels as Kernels>::DeltaNetPrefillPrepKernel,
    cumsum: DeltaNetChunkedCumsumVulkanKernel,
    gram: DeltaNetChunkedGramVulkanKernel,
    a_diag_inv: DeltaNetChunkedADiagInvVulkanKernel,
    causal_inv: DeltaNetChunkedCausalInvVulkanKernel,
    output_and_state: DeltaNetChunkedOutputAndStateVulkanKernel,
}

impl DeltaNetChunkedPrefill for VkDeltaNetChunkedPrefill {
    type Backend = Vulkan;

    fn new(
        context: &VkContext,
        outer_data_type: DataType,
        head_dim: u32,
    ) -> Result<Option<Self>, Error> {
        if !matches!(outer_data_type, DataType::F32 | DataType::BF16) || head_dim != HEAD_K_DIM {
            return Ok(None);
        }
        let arc = context.arc();
        Ok(Some(Self {
            prep: DeltaNetPrefillPrepKernel::new(context, outer_data_type, INNER_DATA_TYPE, head_dim, true, false)?,
            cumsum: DeltaNetChunkedCumsumVulkanKernel::new(&arc, CHUNK_SIZE)?,
            gram: DeltaNetChunkedGramVulkanKernel::new(&arc, head_dim, CHUNK_SIZE)?,
            a_diag_inv: DeltaNetChunkedADiagInvVulkanKernel::new(&arc, CHUNK_SIZE)?,
            causal_inv: DeltaNetChunkedCausalInvVulkanKernel::new(&arc, CHUNK_SIZE, VT)?,
            output_and_state: DeltaNetChunkedOutputAndStateVulkanKernel::new(
                &arc,
                outer_data_type,
                outer_data_type,
                head_dim,
                VT,
                false,
            )?,
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
        let num_blocks = CHUNK_SIZE.div_ceil(BLOCK_SIZE);
        let num_col_pairs = num_blocks.div_ceil(2);

        let mut q_norm = command_buffer.allocate_scratch_for_shape(&[suffix_len, args.key_dim], INNER_DATA_TYPE)?;
        let mut k_norm = command_buffer.allocate_scratch_for_shape(&[suffix_len, args.key_dim], INNER_DATA_TYPE)?;
        let mut beta = command_buffer.allocate_scratch_for_shape(&[suffix_len, num_heads], INNER_DATA_TYPE)?;
        let mut log_decay = command_buffer.allocate_scratch_for_shape(&[suffix_len, num_heads], INNER_DATA_TYPE)?;
        let g = command_buffer.allocate_scratch_for_shape(&[num_heads, suffix_len], INNER_DATA_TYPE)?;
        let kk = command_buffer
            .allocate_scratch_for_shape(&[num_chunks, num_groups, CHUNK_SIZE, CHUNK_SIZE], INNER_DATA_TYPE)?;
        let qk = command_buffer
            .allocate_scratch_for_shape(&[num_chunks, num_heads, CHUNK_SIZE, CHUNK_SIZE], INNER_DATA_TYPE)?;
        let a_packed = command_buffer.allocate_scratch_for_shape(
            &[num_chunks, num_heads, num_blocks, num_col_pairs, BLOCK_SIZE, 2 * BLOCK_SIZE],
            INNER_DATA_TYPE,
        )?;
        let a_inv = command_buffer.allocate_scratch_for_shape(
            &[num_chunks, num_heads, num_blocks, BLOCK_SIZE, BLOCK_SIZE],
            INNER_DATA_TYPE,
        )?;
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
        // SAFETY (every stage): the scratch ranges above have Metal's shapes for these sizes, `in_projected` holds
        // [suffix_len, 2 key_dim + 2 value_dim + 2 num_heads] rows, `ssm_state` [num_heads, value_head_dim, 128] and
        // `delta_output` [suffix_len, value_dim]; each stage writes only ranges no other argument of it aliases, and a
        // stage reads only entries an earlier stage wrote.
        unsafe {
            self.cumsum.encode(
                VkBuffer::resolve_ref(&log_decay),
                VkBuffer::resolve_ref(&g),
                num_heads,
                suffix_len,
                command_buffer,
            );
            self.gram.encode(
                VkBuffer::resolve_ref(&q_norm),
                VkBuffer::resolve_ref(&k_norm),
                VkBuffer::resolve_ref(&g),
                VkBuffer::resolve_ref(&kk),
                VkBuffer::resolve_ref(&qk),
                num_heads,
                num_groups,
                args.key_dim,
                suffix_len,
                command_buffer,
            );
            self.a_diag_inv.encode(
                VkBuffer::resolve_ref(&kk),
                VkBuffer::resolve_ref(&beta),
                VkBuffer::resolve_ref(&g),
                VkBuffer::resolve_ref(&a_packed),
                VkBuffer::resolve_ref(&a_inv),
                num_heads,
                num_groups,
                suffix_len,
                command_buffer,
            );
            self.causal_inv.encode(
                VkBuffer::resolve_ref(&a_packed),
                VkBuffer::resolve_ref(&a_inv),
                VkBuffer::resolve_ref(&t_mat),
                num_heads,
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
