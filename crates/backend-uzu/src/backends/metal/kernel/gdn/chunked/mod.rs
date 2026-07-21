use super::super::{
    DeltaNetChunkedADiagInvMetalKernel, DeltaNetChunkedCausalInvMetalKernel, DeltaNetChunkedCumsumMetalKernel,
    DeltaNetChunkedGramMetalKernel, DeltaNetChunkedOutputAndStateMetalKernel, DeltaNetPrefillPrepMetalKernel,
};
use crate::{
    array::size_for_shape,
    backends::{
        common::{
            Backend, Encoder,
            kernel::{
                DeltaNetPrefillPrepKernel,
                delta_net_chunked_prefill::{DeltaNetChunkedPrefill, DeltaNetChunkedPrefillArgs},
            },
        },
        metal::{Metal, MetalContext},
    },
    data_type::DataType,
};

const MXU_MIN_T: usize = 256;
const CHUNK_SIZE: usize = 64;
const BLOCK_SIZE: usize = 16;
const VT: usize = 32;

pub struct MetalDeltaNetChunkedPrefill {
    min_t: usize,
    prep: DeltaNetPrefillPrepMetalKernel,
    cumsum: DeltaNetChunkedCumsumMetalKernel,
    gram: DeltaNetChunkedGramMetalKernel,
    a_diag_inv: DeltaNetChunkedADiagInvMetalKernel,
    causal_inv: DeltaNetChunkedCausalInvMetalKernel,
    output_and_state: DeltaNetChunkedOutputAndStateMetalKernel,
}

impl DeltaNetChunkedPrefill<Metal> for MetalDeltaNetChunkedPrefill {
    fn new(
        context: &MetalContext,
        outer_data_type: DataType,
        head_dim: u32,
    ) -> Result<Option<Self>, <Metal as Backend>::Error> {
        if outer_data_type == DataType::F16 {
            return Ok(None);
        }

        if !context.supports_mxu() {
            return Ok(None);
        }

        let scratch_data_type = if outer_data_type == DataType::BF16 {
            DataType::BF16
        } else {
            DataType::F32
        };

        Ok(Some(Self {
            min_t: MXU_MIN_T,
            prep: DeltaNetPrefillPrepMetalKernel::new(context, outer_data_type, DataType::F32, head_dim, true, false)?,
            cumsum: DeltaNetChunkedCumsumMetalKernel::new(context, CHUNK_SIZE as u32)?,
            gram: DeltaNetChunkedGramMetalKernel::new(context, head_dim, CHUNK_SIZE as u32)?,
            a_diag_inv: DeltaNetChunkedADiagInvMetalKernel::new(context, CHUNK_SIZE as u32)?,
            causal_inv: DeltaNetChunkedCausalInvMetalKernel::new(context, CHUNK_SIZE as u32, VT as u32)?,
            output_and_state: DeltaNetChunkedOutputAndStateMetalKernel::new(
                context,
                outer_data_type,
                outer_data_type,
                scratch_data_type,
                head_dim,
                VT as u32,
                true,
            )?,
        }))
    }

    fn should_use(
        &self,
        suffix_len: usize,
    ) -> bool {
        suffix_len >= self.min_t
    }

    fn encode(
        &self,
        args: DeltaNetChunkedPrefillArgs<'_, Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), <Metal as Backend>::Error> {
        let suffix_len = args.suffix_len;
        let num_chunks = suffix_len.div_ceil(CHUNK_SIZE);
        let num_blocks = CHUNK_SIZE.div_ceil(BLOCK_SIZE);
        let num_col_pairs = num_blocks.div_ceil(2);

        let mut q_norm =
            encoder.allocate_scratch(size_for_shape(&[suffix_len * args.key_dim as usize], DataType::F32))?;
        let mut k_norm =
            encoder.allocate_scratch(size_for_shape(&[suffix_len * args.key_dim as usize], DataType::F32))?;
        let mut beta =
            encoder.allocate_scratch(size_for_shape(&[suffix_len * args.num_heads as usize], DataType::F32))?;
        let mut log_decay =
            encoder.allocate_scratch(size_for_shape(&[suffix_len * args.num_heads as usize], DataType::F32))?;
        let mut g = encoder.allocate_scratch(size_for_shape(&[suffix_len * args.num_heads as usize], DataType::F32))?;
        let mut kk = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * args.num_groups as usize * CHUNK_SIZE * CHUNK_SIZE],
            DataType::F32,
        ))?;
        let mut qk = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * args.num_heads as usize * CHUNK_SIZE * CHUNK_SIZE],
            DataType::F32,
        ))?;
        let mut a_packed = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * args.num_heads as usize * num_blocks * num_col_pairs * BLOCK_SIZE * 2 * BLOCK_SIZE],
            DataType::F32,
        ))?;
        let mut a_inv = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * args.num_heads as usize * num_blocks * BLOCK_SIZE * BLOCK_SIZE],
            DataType::F32,
        ))?;
        let mut t_mat = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * args.num_heads as usize * CHUNK_SIZE * CHUNK_SIZE],
            DataType::BF16,
        ))?;

        self.prep.encode(
            args.in_projected,
            args.a_log,
            args.dt_bias,
            &mut q_norm,
            &mut k_norm,
            None::<&mut crate::backends::common::Allocation<Metal>>,
            &mut beta,
            &mut log_decay,
            args.num_heads,
            args.num_groups,
            args.key_dim,
            args.value_dim,
            suffix_len as u32,
            encoder,
        );
        self.cumsum.encode(&log_decay, &mut g, args.num_heads, suffix_len as u32, encoder);
        self.gram.encode(
            &q_norm,
            &k_norm,
            &g,
            &mut kk,
            &mut qk,
            args.num_heads,
            args.num_groups,
            args.key_dim,
            suffix_len as u32,
            encoder,
        );
        self.a_diag_inv.encode(
            &kk,
            &beta,
            &g,
            &mut a_packed,
            &mut a_inv,
            args.num_heads,
            args.num_groups,
            suffix_len as u32,
            encoder,
        );
        self.causal_inv.encode(&a_packed, &a_inv, &mut t_mat, args.num_heads, suffix_len as u32, encoder);
        self.output_and_state.encode(
            &q_norm,
            &k_norm,
            args.in_projected,
            &qk,
            &t_mat,
            &g,
            &beta,
            args.ssm_state,
            args.delta_output,
            args.num_heads,
            args.num_groups,
            args.value_head_dim,
            args.key_dim,
            args.value_dim,
            suffix_len as u32,
            encoder,
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../../../../tests/unit/backends/metal/kernel/gdn/chunked_test.rs"]
mod tests;
