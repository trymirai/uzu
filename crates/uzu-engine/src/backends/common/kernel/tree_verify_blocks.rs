use crate::{
    backends::common::{
        Backend, CommandBuffer, CommandBufferEncoding, Kernels,
        kernel::{TreeVerifyKernel, delta_net_tree_verify::DeltaNetTreeVerify},
    },
    encodable_block::mixer::delta_net::tree_verify::{TreeVerifyEncodeArguments, TreeVerifyNewArguments},
};

const TOKEN_BLOCK: u32 = 16;
const MAX_TREE_SIZE: u32 = 2 * TOKEN_BLOCK;

// Verifies a tree of up to two token blocks with the TreeVerify kernel specialized for one or two blocks.
pub struct TreeVerifyBlocks<B: Backend> {
    arguments: TreeVerifyNewArguments,
    kernels: [<B::Kernels as Kernels>::TreeVerifyKernel; 2],
}

impl<B: Backend<Kernels: Kernels<DeltaNetTreeVerify = TreeVerifyBlocks<B>>>> DeltaNetTreeVerify
    for TreeVerifyBlocks<B>
{
    type Backend = B;

    fn is_supported(_context: &B::Context) -> bool {
        true
    }

    fn new(
        context: &B::Context,
        arguments: &TreeVerifyNewArguments,
    ) -> Result<Self, B::Error> {
        let new = |blocks| {
            <B::Kernels as Kernels>::TreeVerifyKernel::new(context, arguments.data_type, blocks, arguments.act_type)
        };
        Ok(Self {
            arguments: *arguments,
            kernels: [new(1)?, new(2)?],
        })
    }

    fn encode(
        &self,
        arguments: TreeVerifyEncodeArguments<'_, B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        assert!(arguments.tree_size <= MAX_TREE_SIZE, "GDN tree verification supports at most {MAX_TREE_SIZE} nodes");
        let mut output = command_buffer.allocate_scratch_for_shape(
            &[arguments.tree_size, self.arguments.num_v_heads, self.arguments.head_v_dim],
            self.arguments.data_type,
        )?;
        self.kernels[(arguments.tree_size > TOKEN_BLOCK) as usize].encode(
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
            arguments.activation_alpha,
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
