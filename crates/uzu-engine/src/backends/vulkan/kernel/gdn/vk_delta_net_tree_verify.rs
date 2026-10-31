use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding,
            kernel::{DeltaNetNormGateKernel, delta_net_tree_verify::DeltaNetTreeVerify},
        },
        vulkan::{
            DeltaNetNormGateVulkanKernel, Error, TreeVerifyValuesVulkanKernel, VkBuffer, VkCommandBufferEncoding,
            VkContext, Vulkan, gpu_types::delta_net::TREE_VERIFY_TOKEN_BLOCK,
        },
    },
    encodable_block::mixer::delta_net::tree_verify::{TreeVerifyEncodeArguments, TreeVerifyNewArguments},
};

/// VkTreeVerify's TreeVerifyValues and DeltaNetNormGate, whose values kernel can also advance the state first along the
/// previous tree's accepted path, which spares the accept a StateAdvance and its pass over the state.
pub struct VkDeltaNetTreeVerify {
    arguments: TreeVerifyNewArguments,
    /// For trees of up to one and two token blocks, from the state as it is and after advancing it.
    values: [[TreeVerifyValuesVulkanKernel; 2]; 2],
    norm_gate: DeltaNetNormGateVulkanKernel,
}

impl DeltaNetTreeVerify for VkDeltaNetTreeVerify {
    type Backend = Vulkan;

    fn is_supported(_context: &VkContext) -> bool {
        true
    }

    fn new(
        context: &VkContext,
        arguments: &TreeVerifyNewArguments,
    ) -> Result<Self, Error> {
        let values = |advance| -> Result<_, Error> {
            Ok([
                TreeVerifyValuesVulkanKernel::new(&context.arc(), arguments.data_type, 1, advance)?,
                TreeVerifyValuesVulkanKernel::new(&context.arc(), arguments.data_type, 2, advance)?,
            ])
        };
        Ok(Self {
            arguments: *arguments,
            values: [values(false)?, values(true)?],
            norm_gate: DeltaNetNormGateVulkanKernel::new(context, arguments.data_type)?,
        })
    }

    fn encode(
        &self,
        arguments: TreeVerifyEncodeArguments<'_, Vulkan>,
        command_buffer: &mut VkCommandBufferEncoding,
    ) -> Result<<Vulkan as Backend>::ScratchBuffer, Error> {
        let mut output = command_buffer.allocate_scratch_for_shape(
            &[arguments.tree_size, self.arguments.num_v_heads, self.arguments.head_v_dim],
            self.arguments.data_type,
        )?;
        let path = arguments.advance.map(|(path, _)| path);
        let accepted_indices =
            path.map(|path| command_buffer.allocate_constant_from_slice(&path.indices)).transpose()?;
        let state = VkBuffer::resolve_ref(&*arguments.h0);
        let (h0, state) = if path.is_some() {
            (None, Some(state))
        } else {
            (Some(state), None)
        };
        // SAFETY: the common contract passes [tree, heads, 128] keys, queries, values and output, [tree] nodes,
        // [tree, value heads] decays and betas, a [value heads, 128, 128] state and an accepted path of nodes of the
        // previous tree, whose tensors have the same rows; the kernel indexes only within those shapes.
        unsafe {
            self.values[path.is_some() as usize][(arguments.tree_size > TREE_VERIFY_TOKEN_BLOCK) as usize].encode(
                VkBuffer::resolve_ref(arguments.q),
                VkBuffer::resolve_ref(arguments.k),
                VkBuffer::resolve_ref(arguments.v),
                VkBuffer::resolve_ref(arguments.trie),
                VkBuffer::resolve_ref(arguments.log_decay),
                VkBuffer::resolve_ref(arguments.beta),
                h0,
                state,
                path.map(|path| VkBuffer::resolve_ref(&path.k)),
                path.map(|path| VkBuffer::resolve_ref(&path.v)),
                path.map(|path| VkBuffer::resolve_ref(&path.log_decay)),
                path.map(|path| VkBuffer::resolve_ref(&path.beta)),
                accepted_indices.as_ref().map(VkBuffer::resolve_ref),
                VkBuffer::resolve_ref(&output),
                1.0,
                1,
                arguments.tree_size,
                self.arguments.num_k_heads,
                self.arguments.num_v_heads,
                self.arguments.head_k_dim,
                self.arguments.head_v_dim,
                path.map(|path| path.indices.len() as u32),
                command_buffer,
            );
        }
        self.norm_gate.encode(
            &mut output,
            arguments.in_proj,
            arguments.norm_weight,
            self.arguments.num_v_heads,
            self.arguments.head_v_dim,
            self.arguments.num_v_heads * self.arguments.head_v_dim,
            arguments.conv_dim,
            arguments.total_proj_dim,
            arguments.norm_epsilon,
            arguments.tree_size,
            command_buffer,
        );
        Ok(output)
    }
}
