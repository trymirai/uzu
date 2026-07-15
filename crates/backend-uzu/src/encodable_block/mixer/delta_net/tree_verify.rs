use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        kernel::{
            BuildTreeGramKernel, BuildTreeOutKernel, BuildTreePrefixKernel, TreeUpdateSolveKernel,
            delta_net_tree_verify::DeltaNetTreeVerify as DeltaNetTreeVerifyTrait,
        },
    },
    data_type::DataType,
};

pub const TOKEN_BLOCK: usize = 16;
pub const BLOCK_PAIR_WIDTH: usize = 2 * TOKEN_BLOCK;

#[derive(Clone, Copy)]
pub struct TreeVerifyNewArguments {
    pub data_type: DataType,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
}

#[derive(Clone, Copy)]
pub struct TreeVerifyEncodeArguments<'a, B: Backend> {
    pub q: &'a Allocation<B>,
    pub k: &'a Allocation<B>,
    pub v: &'a Allocation<B>,
    pub trie: &'a Allocation<B>,
    pub log_decay: &'a Allocation<B>,
    pub beta: &'a Allocation<B>,
    pub h0: &'a Allocation<B>,
    pub tree_size: usize,
}

struct Layout {
    tree_size: usize,
    num_blocks: usize,
    num_block_pairs: usize,
    num_v_heads: usize,
    head_v_dim: usize,
}

impl Layout {
    fn new(
        tree_size: usize,
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

    const fn a_packed_shape(&self) -> [usize; 5] {
        [self.num_v_heads, self.num_blocks, self.num_block_pairs, TOKEN_BLOCK, BLOCK_PAIR_WIDTH]
    }

    const fn a_inverse_shape(&self) -> [usize; 4] {
        [self.num_v_heads, self.num_blocks, TOKEN_BLOCK, TOKEN_BLOCK]
    }
}

pub(crate) fn encode_with<B: Backend>(
    new_arguments: &TreeVerifyNewArguments,
    prefix_kernel: &<B::Kernels as Kernels>::BuildTreePrefixKernel,
    gram_kernel: &<B::Kernels as Kernels>::BuildTreeGramKernel,
    solve_kernel: &<B::Kernels as Kernels>::TreeUpdateSolveKernel,
    out_kernel: &<B::Kernels as Kernels>::BuildTreeOutKernel,
    arguments: TreeVerifyEncodeArguments<'_, B>,
    encoder: &mut Encoder<B>,
) -> Result<Allocation<B>, B::Error> {
    let layout = Layout::new(arguments.tree_size, new_arguments);
    let mut h0_indices = encoder.allocate_constant(DataType::I32.size_in_bytes())?;
    h0_indices.copyin(&[0i32]);
    let mut scratch = |shape: &[usize], data_type| encoder.allocate_scratch(size_for_shape(shape, data_type));
    let mut prefix = scratch(&[layout.tree_size, layout.num_v_heads], DataType::F32)?;
    let mut a_packed = scratch(&layout.a_packed_shape(), DataType::F32)?;
    let mut qkd = scratch(&[layout.num_v_heads, layout.tree_size, layout.tree_size], DataType::F32)?;
    let mut a_inverse = scratch(&layout.a_inverse_shape(), DataType::F32)?;
    let mut kh0 = scratch(&[layout.tree_size, layout.num_v_heads, layout.head_v_dim], DataType::F32)?;
    let mut u = scratch(&[layout.num_v_heads, layout.tree_size, layout.head_v_dim], DataType::F32)?;
    let mut output = scratch(&[layout.tree_size, layout.num_v_heads, layout.head_v_dim], new_arguments.data_type)?;
    let tree_size = arguments.tree_size.try_into().expect("tree size exceeds u32");
    let num_k_heads = new_arguments.num_k_heads.try_into().expect("K head count exceeds u32");
    let num_v_heads = new_arguments.num_v_heads.try_into().expect("V head count exceeds u32");
    let head_k_dim = new_arguments.head_k_dim.try_into().expect("K head dimension exceeds u32");
    let head_v_dim = new_arguments.head_v_dim.try_into().expect("V head dimension exceeds u32");

    prefix_kernel.encode(arguments.trie, arguments.log_decay, &mut prefix, 1, tree_size, num_v_heads, encoder);
    gram_kernel.encode(
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
        tree_size,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        encoder,
    );
    solve_kernel.encode(
        Some(&kh0),
        arguments.v,
        &prefix,
        arguments.beta,
        &a_packed,
        &a_inverse,
        Some(&h0_indices),
        &mut u,
        1,
        tree_size,
        num_v_heads,
        head_v_dim,
        encoder,
    );
    out_kernel.encode(
        arguments.q,
        &prefix,
        &qkd,
        &u,
        Some(arguments.h0),
        Some(&h0_indices),
        &mut output,
        1.0,
        1,
        tree_size,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        encoder,
    );
    Ok(output)
}

struct TreeVerifyFallback<B: Backend> {
    arguments: TreeVerifyNewArguments,
    prefix: <B::Kernels as Kernels>::BuildTreePrefixKernel,
    gram: <B::Kernels as Kernels>::BuildTreeGramKernel,
    solve: <B::Kernels as Kernels>::TreeUpdateSolveKernel,
    out: <B::Kernels as Kernels>::BuildTreeOutKernel,
}

impl<B: Backend> TreeVerifyFallback<B> {
    fn new(
        arguments: TreeVerifyNewArguments,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            arguments,
            prefix: <B::Kernels as Kernels>::BuildTreePrefixKernel::new(context)?,
            gram: <B::Kernels as Kernels>::BuildTreeGramKernel::new(context, arguments.data_type, false, true)?,
            solve: <B::Kernels as Kernels>::TreeUpdateSolveKernel::new(context, arguments.data_type, 32, true)?,
            out: <B::Kernels as Kernels>::BuildTreeOutKernel::new(
                context,
                arguments.data_type,
                arguments.data_type,
                false,
                false,
                true,
            )?,
        })
    }

    fn encode(
        &self,
        arguments: TreeVerifyEncodeArguments<'_, B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        encode_with(&self.arguments, &self.prefix, &self.gram, &self.solve, &self.out, arguments, encoder)
    }
}

pub struct TreeVerifyCores<B: Backend> {
    core: TreeVerifyCore<B>,
}

enum TreeVerifyCore<B: Backend> {
    Optimized(<B::Kernels as Kernels>::DeltaNetTreeVerify),
    Fallback(TreeVerifyFallback<B>),
}

impl<B: Backend> TreeVerifyCores<B> {
    pub fn new(
        arguments: TreeVerifyNewArguments,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        let core = if <<B::Kernels as Kernels>::DeltaNetTreeVerify as DeltaNetTreeVerifyTrait<B>>::is_supported(
            &arguments, context,
        )? {
            TreeVerifyCore::Optimized(<<B::Kernels as Kernels>::DeltaNetTreeVerify as DeltaNetTreeVerifyTrait<B>>::new(
                context, &arguments,
            )?)
        } else {
            TreeVerifyCore::Fallback(TreeVerifyFallback::new(arguments, context)?)
        };
        Ok(Self {
            core,
        })
    }

    pub fn encode(
        &self,
        arguments: TreeVerifyEncodeArguments<'_, B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        match &self.core {
            TreeVerifyCore::Optimized(core) => core.encode(arguments, encoder),
            TreeVerifyCore::Fallback(core) => core.encode(arguments, encoder),
        }
    }
}

#[cfg(test)]
#[path = "../../../../unit/encodable_block/delta_net_tree_verify_bench.rs"]
mod tests;
