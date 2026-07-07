use std::any::Any;

use super::{DeltaNet, DeltaNetState};
use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Context, Encoder, Kernels,
        kernel::{
            DeltaNetChunkedCumsumKernel, DeltaNetChunkedGramKernel, DeltaNetChunkedMegaApplyKernel,
            DeltaNetChunkedPrepKernel, DeltaNetChunkedSolveKernel, DeltaNetChunkedSolveTKernel,
        },
    },
    data_type::DataType,
};
#[cfg(metal_backend)]
use crate::backends::metal::MetalContext;

const MXU_MIN_T: usize = 256;
const SIMD_MIN_T: usize = 1024;
const CHUNK_SIZE: usize = 64;
const BLOCK_SIZE: usize = 16;
const VT: usize = 32;

pub(super) struct ChunkedPrefill<B: Backend> {
    min_t: usize,
    prep: <B::Kernels as Kernels>::DeltaNetChunkedPrepKernel,
    cumsum: <B::Kernels as Kernels>::DeltaNetChunkedCumsumKernel,
    gram: <B::Kernels as Kernels>::DeltaNetChunkedGramKernel,
    solve: <B::Kernels as Kernels>::DeltaNetChunkedSolveKernel,
    solve_t: <B::Kernels as Kernels>::DeltaNetChunkedSolveTKernel,
    mega: <B::Kernels as Kernels>::DeltaNetChunkedMegaApplyKernel,
}

impl<B: Backend> ChunkedPrefill<B> {
    pub(super) fn new(
        context: &B::Context,
        outer_data_type: DataType,
        head_dim: u32,
    ) -> Result<Option<Self>, B::Error>
    where
        B::Context: Any,
    {
        let Some((use_mxu, min_t)) = chunked_prefill_config(context) else {
            return Ok(None);
        };

        Ok(Some(Self {
            min_t,
            prep: <B::Kernels as Kernels>::DeltaNetChunkedPrepKernel::new(context, outer_data_type, head_dim)?,
            cumsum: <B::Kernels as Kernels>::DeltaNetChunkedCumsumKernel::new(context)?,
            gram: <B::Kernels as Kernels>::DeltaNetChunkedGramKernel::new(context, head_dim, CHUNK_SIZE as u32)?,
            solve: <B::Kernels as Kernels>::DeltaNetChunkedSolveKernel::new(context, CHUNK_SIZE as u32, false)?,
            solve_t: <B::Kernels as Kernels>::DeltaNetChunkedSolveTKernel::new(context, CHUNK_SIZE as u32, VT as u32)?,
            mega: <B::Kernels as Kernels>::DeltaNetChunkedMegaApplyKernel::new(
                context,
                outer_data_type,
                outer_data_type,
                VT as u32,
                use_mxu,
            )?,
        }))
    }

    pub(super) fn should_use(
        &self,
        suffix_len: usize,
    ) -> bool {
        suffix_len >= self.min_t
    }

    pub(super) fn encode(
        &self,
        delta_net: &DeltaNet<B>,
        in_projected: &Allocation<B>,
        state: &mut DeltaNetState<B>,
        delta_output: &mut Allocation<B>,
        suffix_len: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let num_chunks = suffix_len.div_ceil(CHUNK_SIZE);
        let num_blocks = CHUNK_SIZE.div_ceil(BLOCK_SIZE);
        let num_col_pairs = num_blocks.div_ceil(2);

        let mut q_norm = encoder.allocate_scratch(size_for_shape(&[suffix_len * delta_net.key_dim], DataType::F32))?;
        let mut k_norm = encoder.allocate_scratch(size_for_shape(&[suffix_len * delta_net.key_dim], DataType::F32))?;
        let mut beta = encoder.allocate_scratch(size_for_shape(&[suffix_len * delta_net.num_heads], DataType::F32))?;
        let mut log_decay =
            encoder.allocate_scratch(size_for_shape(&[suffix_len * delta_net.num_heads], DataType::F32))?;
        let mut g = encoder.allocate_scratch(size_for_shape(&[suffix_len * delta_net.num_heads], DataType::F32))?;
        let mut kk = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * delta_net.num_groups * CHUNK_SIZE * CHUNK_SIZE],
            DataType::F32,
        ))?;
        let mut qk = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * delta_net.num_heads * CHUNK_SIZE * CHUNK_SIZE],
            DataType::F32,
        ))?;
        let mut a_packed = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * delta_net.num_heads * num_blocks * num_col_pairs * BLOCK_SIZE * 2 * BLOCK_SIZE],
            DataType::F32,
        ))?;
        let mut a_inv = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * delta_net.num_heads * num_blocks * BLOCK_SIZE * BLOCK_SIZE],
            DataType::F32,
        ))?;
        let mut t_mat = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * delta_net.num_heads * CHUNK_SIZE * CHUNK_SIZE],
            DataType::BF16,
        ))?;

        self.prep.encode(
            in_projected,
            &delta_net.a_log,
            &delta_net.dt_bias,
            &mut q_norm,
            &mut k_norm,
            &mut beta,
            &mut log_decay,
            delta_net.num_heads as u32,
            delta_net.num_groups as u32,
            delta_net.key_dim as u32,
            delta_net.value_dim as u32,
            suffix_len as u32,
            encoder,
        );
        self.cumsum.encode(
            &log_decay,
            &mut g,
            delta_net.num_heads as u32,
            suffix_len as u32,
            CHUNK_SIZE as u32,
            encoder,
        );
        self.gram.encode(
            &q_norm,
            &k_norm,
            &g,
            &mut kk,
            &mut qk,
            delta_net.num_heads as u32,
            delta_net.num_groups as u32,
            delta_net.key_dim as u32,
            suffix_len as u32,
            encoder,
        );
        self.solve.encode(
            &kk,
            &beta,
            &g,
            &mut a_packed,
            &mut a_inv,
            delta_net.num_heads as u32,
            delta_net.num_groups as u32,
            suffix_len as u32,
            encoder,
        );
        self.solve_t.encode(&a_packed, &a_inv, &mut t_mat, delta_net.num_heads as u32, suffix_len as u32, encoder);
        self.mega.encode(
            &q_norm,
            &k_norm,
            in_projected,
            &qk,
            &t_mat,
            &g,
            &beta,
            &mut state.ssm_state,
            delta_output,
            delta_net.num_heads as u32,
            delta_net.num_groups as u32,
            delta_net.value_head_dim as u32,
            delta_net.key_dim as u32,
            delta_net.value_dim as u32,
            suffix_len as u32,
            encoder,
        );
        Ok(())
    }
}

fn chunked_prefill_config<C: Any>(context: &C) -> Option<(bool, usize)> {
    #[cfg(metal_backend)]
    if let Some(context) = (context as &dyn Any).downcast_ref::<MetalContext>() {
        let use_mxu = context.supports_mxu();
        return if use_mxu {
            Some((true, MXU_MIN_T))
        } else if context.supports_dynamic_caching() {
            Some((false, SIMD_MIN_T))
        } else {
            None
        };
    }

    #[cfg(not(metal_backend))]
    let _ = context;

    None
}
