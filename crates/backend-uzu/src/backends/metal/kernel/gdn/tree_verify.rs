use crate::{
    array::size_for_shape,
    backends::{
        common::{
            Allocation, Encoder, Kernels,
            kernel::{
                BuildTreeGramKernel, BuildTreeOutKernel, BuildTreePrefixKernel, TreeUpdateSolveKernel,
                delta_net_tree_verify::DeltaNetTreeVerify,
            },
        },
        metal::{Metal, MetalContext, device_tier::DeviceTier, error::MetalError, kernel::MetalKernels},
    },
    data_type::DataType,
    encodable_block::mixer::delta_net::tree_verify::{TreeVerifyEncodeArguments, TreeVerifyNewArguments},
};

const TOKEN_BLOCK: usize = 16;
const BLOCK_PAIR_WIDTH: usize = 2 * TOKEN_BLOCK;

struct Layout {
    tree_size: usize,
    num_blocks: usize,
    num_block_pairs: usize,
    num_v_heads: usize,
    head_v_dim: usize,
}

impl Layout {
    const fn new(
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

pub struct MetalDeltaNetTreeVerify {
    arguments: TreeVerifyNewArguments,
    prefix: <MetalKernels as Kernels>::BuildTreePrefixKernel,
    gram: <MetalKernels as Kernels>::BuildTreeGramKernel,
    solve: <MetalKernels as Kernels>::TreeUpdateSolveKernel,
    out: <MetalKernels as Kernels>::BuildTreeOutKernel,
}

impl DeltaNetTreeVerify<Metal> for MetalDeltaNetTreeVerify {
    fn is_supported(_context: &MetalContext) -> bool {
        true
    }

    fn new(
        context: &MetalContext,
        arguments: &TreeVerifyNewArguments,
    ) -> Result<Self, MetalError> {
        let use_mxu = arguments.data_type == DataType::BF16 && context.supports_mxu();
        let transposed_h0 =
            !use_mxu && matches!(context.device_tier(), DeviceTier::SmallLegacy | DeviceTier::SmallApple8);
        Ok(Self {
            arguments: *arguments,
            prefix: <MetalKernels as Kernels>::BuildTreePrefixKernel::new(context)?,
            gram: <MetalKernels as Kernels>::BuildTreeGramKernel::new(context, arguments.data_type, use_mxu, true)?,
            solve: <MetalKernels as Kernels>::TreeUpdateSolveKernel::new(context, arguments.data_type, 32, true)?,
            out: <MetalKernels as Kernels>::BuildTreeOutKernel::new(
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
        arguments: TreeVerifyEncodeArguments<'_, Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<Allocation<Metal>, MetalError> {
        let layout = Layout::new(arguments.tree_size, &self.arguments);
        let mut h0_indices = encoder.allocate_constant(DataType::I32.size_in_bytes())?;
        h0_indices.copyin(&[0i32]);
        let mut scratch = |shape: &[usize], data_type| encoder.allocate_scratch(size_for_shape(shape, data_type));
        let mut prefix = scratch(&[layout.tree_size, layout.num_v_heads], DataType::F32)?;
        let mut a_packed = scratch(&layout.a_packed_shape(), DataType::F32)?;
        let mut qkd = scratch(&[layout.num_v_heads, layout.tree_size, layout.tree_size], DataType::F32)?;
        let mut a_inverse = scratch(&layout.a_inverse_shape(), DataType::F32)?;
        let mut kh0 = scratch(&[layout.tree_size, layout.num_v_heads, layout.head_v_dim], DataType::F32)?;
        let mut u = scratch(&[layout.num_v_heads, layout.tree_size, layout.head_v_dim], DataType::F32)?;
        let mut output = scratch(&[layout.tree_size, layout.num_v_heads, layout.head_v_dim], self.arguments.data_type)?;
        let tree_size = arguments.tree_size.try_into().expect("tree size exceeds u32");
        let num_k_heads = self.arguments.num_k_heads.try_into().expect("K head count exceeds u32");
        let num_v_heads = self.arguments.num_v_heads.try_into().expect("V head count exceeds u32");
        let head_k_dim = self.arguments.head_k_dim.try_into().expect("K head dimension exceeds u32");
        let head_v_dim = self.arguments.head_v_dim.try_into().expect("V head dimension exceeds u32");

        self.prefix.encode(arguments.trie, arguments.log_decay, &mut prefix, 1, tree_size, num_v_heads, encoder);
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
            tree_size,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            encoder,
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
            tree_size,
            num_v_heads,
            head_v_dim,
            encoder,
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
            tree_size,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            encoder,
        );
        Ok(output)
    }
}
