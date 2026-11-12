use crate::{
    backends::{
        common::{
            BufferMut, BufferRef,
            kernel::{DeltaNetNormGateKernel, TreeVerifyKernel},
        },
        vulkan::{
            DeltaNetNormGateVulkanKernel, Error, TreeVerifyValuesVulkanKernel, VkBuffer, VkCommandBufferEncoding,
            VkContext, Vulkan,
        },
    },
    data_type::DataType,
};

/// GDN tree verification as TreeVerifyValues, a workgroup per 32-value tile of each head, then DeltaNetNormGate on its
/// output: the norm spans a head's 128 values, which one workgroup per head would leave most SMs idle to reach.
pub struct VkTreeVerify {
    values: TreeVerifyValuesVulkanKernel,
    norm_gate: DeltaNetNormGateVulkanKernel,
}

impl TreeVerifyKernel for VkTreeVerify {
    type Backend = Vulkan;

    fn new(
        context: &VkContext,
        data_type: DataType,
        token_blocks: u32,
    ) -> Result<Self, Error> {
        Ok(Self {
            values: TreeVerifyValuesVulkanKernel::new(&context.arc(), data_type, token_blocks)?,
            norm_gate: DeltaNetNormGateVulkanKernel::new(context, data_type)?,
        })
    }

    fn encode(
        &self,
        q: impl BufferRef<Backend = Vulkan>,
        k: impl BufferRef<Backend = Vulkan>,
        v: impl BufferRef<Backend = Vulkan>,
        trie: impl BufferRef<Backend = Vulkan>,
        log_decay: impl BufferRef<Backend = Vulkan>,
        beta: impl BufferRef<Backend = Vulkan>,
        h0: impl BufferRef<Backend = Vulkan>,
        in_proj: impl BufferRef<Backend = Vulkan>,
        norm_weight: impl BufferRef<Backend = Vulkan>,
        mut o: impl BufferMut<Backend = Vulkan>,
        scale: f32,
        norm_epsilon: f32,
        batch_size: u32,
        tree_size: u32,
        num_k_heads: u32,
        num_v_heads: u32,
        head_k_dim: u32,
        head_v_dim: u32,
        conv_dim: u32,
        total_proj_dim: u32,
        command_buffer: &mut VkCommandBufferEncoding,
    ) {
        let (output, output_range) = o.reborrow().parts();
        // SAFETY: the common contract passes [batch, tree, heads, 128] keys, queries, values and output, [batch, tree]
        // nodes, [batch, tree, value heads] decays and betas and [batch, value heads, 128, 128] states, which the kernel
        // indexes only within those shapes.
        unsafe {
            self.values.encode(
                VkBuffer::resolve_ref(q),
                VkBuffer::resolve_ref(k),
                VkBuffer::resolve_ref(v),
                VkBuffer::resolve_ref(trie),
                VkBuffer::resolve_ref(log_decay),
                VkBuffer::resolve_ref(beta),
                VkBuffer::resolve_ref(h0),
                VkBuffer::resolve(output, output_range),
                scale,
                batch_size,
                tree_size,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                command_buffer,
            );
        }
        self.norm_gate.encode(
            o,
            in_proj,
            norm_weight,
            num_v_heads,
            head_v_dim,
            num_v_heads * head_v_dim,
            conv_dim,
            total_proj_dim,
            norm_epsilon,
            batch_size * tree_size,
            command_buffer,
        );
    }
}
