use super::{Error, VkCommandBufferEncoding, VkContext, Vulkan, VulkanKernels};
use crate::{
    backends::common::{
        Backend, CommandBufferEncoding, Kernels,
        kernel::{
            BuildTreeGramKernel, BuildTreeOutKernel, BuildTreePrefixKernel, TreeUpdateSolveKernel,
            delta_net_tree_verify::DeltaNetTreeVerify,
        },
    },
    data_type::DataType,
    encodable_block::mixer::delta_net::tree_verify::{TreeVerifyEncodeArguments, TreeVerifyNewArguments},
};

const TOKEN_BLOCK: u32 = 16;
const INNER_DATA_TYPE: DataType = DataType::F32;

pub struct VkDeltaNetTreeVerify {
    arguments: TreeVerifyNewArguments,
    prefix: <VulkanKernels as Kernels>::BuildTreePrefixKernel,
    gram: <VulkanKernels as Kernels>::BuildTreeGramKernel,
    solve: <VulkanKernels as Kernels>::TreeUpdateSolveKernel,
    out: <VulkanKernels as Kernels>::BuildTreeOutKernel,
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
            prefix: BuildTreePrefixKernel::new(context)?,
            gram: BuildTreeGramKernel::new(context, arguments.data_type, false, true)?,
            solve: TreeUpdateSolveKernel::new(context, arguments.data_type, 32, true)?,
            out: BuildTreeOutKernel::new(context, arguments.data_type, arguments.data_type, false, false, true)?,
        })
    }

    fn encode(
        &self,
        arguments: TreeVerifyEncodeArguments<'_, Vulkan>,
        command_buffer: &mut VkCommandBufferEncoding,
    ) -> Result<<Vulkan as Backend>::ScratchBuffer, Error> {
        let tree_size = arguments.tree_size;
        let num_v_heads = self.arguments.num_v_heads;
        let head_v_dim = self.arguments.head_v_dim;
        let num_blocks = tree_size.div_ceil(TOKEN_BLOCK);

        let mut prefix = command_buffer.allocate_scratch_for_shape(&[tree_size, num_v_heads], INNER_DATA_TYPE)?;
        let mut a_packed = command_buffer.allocate_scratch_for_shape(
            &[num_v_heads, num_blocks, num_blocks.div_ceil(2), TOKEN_BLOCK, 2 * TOKEN_BLOCK],
            INNER_DATA_TYPE,
        )?;
        let mut qkd =
            command_buffer.allocate_scratch_for_shape(&[num_v_heads, tree_size, tree_size], INNER_DATA_TYPE)?;
        let mut a_inverse = command_buffer
            .allocate_scratch_for_shape(&[num_v_heads, num_blocks, TOKEN_BLOCK, TOKEN_BLOCK], INNER_DATA_TYPE)?;
        let mut kh0 =
            command_buffer.allocate_scratch_for_shape(&[tree_size, num_v_heads, head_v_dim], INNER_DATA_TYPE)?;
        let mut u =
            command_buffer.allocate_scratch_for_shape(&[num_v_heads, tree_size, head_v_dim], INNER_DATA_TYPE)?;
        let mut output = command_buffer
            .allocate_scratch_for_shape(&[tree_size, num_v_heads, head_v_dim], self.arguments.data_type)?;

        BuildTreePrefixKernel::encode(
            &self.prefix,
            arguments.trie,
            arguments.log_decay,
            &mut prefix,
            1,
            tree_size,
            num_v_heads,
            command_buffer,
        );
        BuildTreeGramKernel::encode(
            &self.gram,
            arguments.q,
            arguments.k,
            arguments.trie,
            &prefix,
            arguments.beta,
            Some(arguments.h0),
            &mut a_packed,
            &mut qkd,
            &mut a_inverse,
            Some(&mut kh0),
            1.0,
            1,
            tree_size,
            self.arguments.num_k_heads,
            num_v_heads,
            self.arguments.head_k_dim,
            head_v_dim,
            command_buffer,
        );
        TreeUpdateSolveKernel::encode(
            &self.solve,
            Some(&kh0),
            arguments.v,
            &prefix,
            arguments.beta,
            &a_packed,
            &a_inverse,
            &mut u,
            1,
            tree_size,
            num_v_heads,
            head_v_dim,
            command_buffer,
        );
        BuildTreeOutKernel::encode(
            &self.out,
            arguments.q,
            &prefix,
            &qkd,
            &u,
            Some(arguments.h0),
            &mut output,
            1.0,
            1,
            tree_size,
            self.arguments.num_k_heads,
            num_v_heads,
            self.arguments.head_k_dim,
            head_v_dim,
            command_buffer,
        );
        Ok(output)
    }
}
