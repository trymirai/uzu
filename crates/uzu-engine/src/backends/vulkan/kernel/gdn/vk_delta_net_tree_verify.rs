use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, Kernels,
            kernel::{TreeVerifyKernel, delta_net_tree_verify::DeltaNetTreeVerify},
        },
        vulkan::{
            Error, VkCommandBufferEncoding, VkContext, Vulkan, VulkanKernels,
            gpu_types::delta_net::TREE_VERIFY_TOKEN_BLOCK,
        },
    },
    encodable_block::mixer::delta_net::tree_verify::{TreeVerifyEncodeArguments, TreeVerifyNewArguments},
};

const MAX_TREE_SIZE: u32 = 2 * TREE_VERIFY_TOKEN_BLOCK;

pub struct VkDeltaNetTreeVerify {
    arguments: TreeVerifyNewArguments,
    /// For trees of up to one and two token blocks.
    kernels: [<VulkanKernels as Kernels>::TreeVerifyKernel; 2],
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
        Ok(Self {
            arguments: *arguments,
            kernels: [
                TreeVerifyKernel::new(context, arguments.data_type, 1)?,
                TreeVerifyKernel::new(context, arguments.data_type, 2)?,
            ],
        })
    }

    fn encode(
        &self,
        arguments: TreeVerifyEncodeArguments<'_, Vulkan>,
        command_buffer: &mut VkCommandBufferEncoding,
    ) -> Result<<Vulkan as Backend>::ScratchBuffer, Error> {
        assert!(arguments.tree_size <= MAX_TREE_SIZE, "GDN tree verification supports at most {MAX_TREE_SIZE} nodes");
        let mut output = command_buffer.allocate_scratch_for_shape(
            &[arguments.tree_size, self.arguments.num_v_heads, self.arguments.head_v_dim],
            self.arguments.data_type,
        )?;
        self.kernels[(arguments.tree_size > TREE_VERIFY_TOKEN_BLOCK) as usize].encode(
            arguments.q,
            arguments.k,
            arguments.v,
            arguments.trie,
            arguments.log_decay,
            arguments.beta,
            arguments.h0,
            arguments.in_proj,
            arguments.norm_weight,
            &mut output,
            1.0,
            arguments.norm_epsilon,
            1,
            arguments.tree_size,
            self.arguments.num_k_heads,
            self.arguments.num_v_heads,
            self.arguments.head_k_dim,
            self.arguments.head_v_dim,
            arguments.conv_dim,
            arguments.total_proj_dim,
            command_buffer,
        );
        Ok(output)
    }
}
