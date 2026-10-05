//! DeltaNet tree verification, as in `backends/metal/kernel/gdn/tree_verify.rs`: prefix decay,
//! Gram / triangular inverse, update solve and output over the draft tree. The fragment math runs on
//! the 8x8 simdgroup path (no MXU on AMD); `transposed_h0` stays off until it is profiled.

use super::AmdgpuKernels;
use crate::{
    backends::{
        amdgpu::{Amdgpu, command_buffer::AmdgpuCommandBufferEncoding, context::AmdgpuContext, error::AmdgpuError},
        common::{
            Backend, CommandBufferEncoding, Kernels,
            kernel::{
                BuildTreeGramKernel, BuildTreeOutKernel, BuildTreePrefixKernel, TreeUpdateSolveKernel,
                delta_net_tree_verify::DeltaNetTreeVerify,
            },
        },
    },
    data_type::DataType,
    encodable_block::mixer::delta_net::tree_verify::{TreeVerifyEncodeArguments, TreeVerifyNewArguments},
};

const TOKEN_BLOCK: u32 = 16;
const BLOCK_PAIR_WIDTH: u32 = 2 * TOKEN_BLOCK;
const INNER_DATA_TYPE: DataType = DataType::F32;

struct Layout {
    tree_size: u32,
    num_blocks: u32,
    num_block_pairs: u32,
    num_v_heads: u32,
    head_v_dim: u32,
}

impl Layout {
    const fn new(
        tree_size: u32,
        arguments: &TreeVerifyNewArguments,
    ) -> Self {
        let num_blocks = tree_size.div_ceil(TOKEN_BLOCK);
        Self {
            tree_size,
            num_blocks,
            num_block_pairs: num_blocks.div_ceil(2),
            num_v_heads: arguments.num_v_heads,
            head_v_dim: arguments.head_v_dim,
        }
    }

    const fn a_packed_shape(&self) -> [u32; 5] {
        [self.num_v_heads, self.num_blocks, self.num_block_pairs, TOKEN_BLOCK, BLOCK_PAIR_WIDTH]
    }

    const fn a_inverse_shape(&self) -> [u32; 4] {
        [self.num_v_heads, self.num_blocks, TOKEN_BLOCK, TOKEN_BLOCK]
    }
}

pub struct AmdgpuDeltaNetTreeVerify {
    arguments: TreeVerifyNewArguments,
    prefix: <AmdgpuKernels as Kernels>::BuildTreePrefixKernel,
    gram: <AmdgpuKernels as Kernels>::BuildTreeGramKernel,
    solve: <AmdgpuKernels as Kernels>::TreeUpdateSolveKernel,
    out: <AmdgpuKernels as Kernels>::BuildTreeOutKernel,
}

impl DeltaNetTreeVerify for AmdgpuDeltaNetTreeVerify {
    type Backend = Amdgpu;

    fn is_supported(_context: &AmdgpuContext) -> bool {
        true
    }

    fn new(
        context: &AmdgpuContext,
        arguments: &TreeVerifyNewArguments,
    ) -> Result<Self, AmdgpuError> {
        let use_mxu = false;
        let transposed_h0 = false;
        Ok(Self {
            arguments: *arguments,
            prefix: <AmdgpuKernels as Kernels>::BuildTreePrefixKernel::new(context)?,
            gram: <AmdgpuKernels as Kernels>::BuildTreeGramKernel::new(context, arguments.data_type, use_mxu, true)?,
            solve: <AmdgpuKernels as Kernels>::TreeUpdateSolveKernel::new(context, arguments.data_type, 32, true)?,
            out: <AmdgpuKernels as Kernels>::BuildTreeOutKernel::new(
                context,
                arguments.data_type,
                arguments.data_type,
                use_mxu,
                transposed_h0,
                true,
            )?,
        })
    }

    fn encode(
        &self,
        arguments: TreeVerifyEncodeArguments<'_, Amdgpu>,
        command_buffer: &mut AmdgpuCommandBufferEncoding,
    ) -> Result<<Amdgpu as Backend>::ScratchBuffer, AmdgpuError> {
        let layout = Layout::new(arguments.tree_size, &self.arguments);
        let mut h0_indices = command_buffer.allocate_scratch(DataType::I32.size_in_bytes())?;
        command_buffer.encode_fill(&mut h0_indices, 0);

        let mut prefix =
            command_buffer.allocate_scratch_for_shape(&[layout.tree_size, layout.num_v_heads], INNER_DATA_TYPE)?;
        let mut a_packed = command_buffer.allocate_scratch_for_shape(&layout.a_packed_shape(), INNER_DATA_TYPE)?;
        let mut qkd = command_buffer
            .allocate_scratch_for_shape(&[layout.num_v_heads, layout.tree_size, layout.tree_size], INNER_DATA_TYPE)?;
        let mut a_inverse = command_buffer.allocate_scratch_for_shape(&layout.a_inverse_shape(), INNER_DATA_TYPE)?;
        let mut kh0 = command_buffer
            .allocate_scratch_for_shape(&[layout.tree_size, layout.num_v_heads, layout.head_v_dim], INNER_DATA_TYPE)?;
        let mut u = command_buffer
            .allocate_scratch_for_shape(&[layout.num_v_heads, layout.tree_size, layout.head_v_dim], INNER_DATA_TYPE)?;
        let mut output = command_buffer.allocate_scratch_for_shape(
            &[layout.tree_size, layout.num_v_heads, layout.head_v_dim],
            self.arguments.data_type,
        )?;

        self.prefix.encode(
            arguments.trie,
            arguments.log_decay,
            &mut prefix,
            1,
            arguments.tree_size,
            self.arguments.num_v_heads,
            command_buffer,
        );
        self.gram.encode(
            arguments.q,
            arguments.k,
            arguments.trie,
            &prefix,
            arguments.beta,
            Some(arguments.h0),
            Some(&h0_indices),
            &mut a_packed,
            &mut qkd,
            &mut a_inverse,
            Some(&mut kh0),
            1.0,
            1,
            arguments.tree_size,
            self.arguments.num_k_heads,
            self.arguments.num_v_heads,
            self.arguments.head_k_dim,
            self.arguments.head_v_dim,
            command_buffer,
        );
        self.solve.encode(
            Some(&kh0),
            arguments.v,
            &prefix,
            arguments.beta,
            &a_packed,
            &a_inverse,
            Some(&h0_indices),
            &mut u,
            1,
            arguments.tree_size,
            self.arguments.num_v_heads,
            self.arguments.head_v_dim,
            command_buffer,
        );
        self.out.encode(
            arguments.q,
            &prefix,
            &qkd,
            &u,
            Some(arguments.h0),
            Some(&h0_indices),
            &mut output,
            1.0,
            1,
            arguments.tree_size,
            self.arguments.num_k_heads,
            self.arguments.num_v_heads,
            self.arguments.head_k_dim,
            self.arguments.head_v_dim,
            command_buffer,
        );
        Ok(output)
    }
}
