use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, Kernels,
            kernel::{TreeVerifyKernel, delta_net_tree_verify::DeltaNetTreeVerify},
        },
        metal::{
            Metal, command_buffer::MetalCommandBufferEncoding, context::MetalContext, error::MetalError,
            kernel::MetalKernels,
        },
    },
    encodable_block::mixer::delta_net::tree_verify::{TreeVerifyEncodeArguments, TreeVerifyNewArguments},
};

const TOKEN_BLOCK: u32 = 16;
const MAX_TREE_SIZE: u32 = 2 * TOKEN_BLOCK;

pub struct MetalDeltaNetTreeVerify {
    arguments: TreeVerifyNewArguments,
    kernels: [<MetalKernels as Kernels>::TreeVerifyKernel; 2],
}

impl DeltaNetTreeVerify for MetalDeltaNetTreeVerify {
    type Backend = Metal;

    fn is_supported(_context: &MetalContext) -> bool {
        true
    }

    fn new(
        context: &MetalContext,
        arguments: &TreeVerifyNewArguments,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            arguments: *arguments,
            kernels: [
                <MetalKernels as Kernels>::TreeVerifyKernel::new(context, arguments.data_type, 1)?,
                <MetalKernels as Kernels>::TreeVerifyKernel::new(context, arguments.data_type, 2)?,
            ],
        })
    }

    fn encode(
        &self,
        arguments: TreeVerifyEncodeArguments<'_, Metal>,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
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
