use super::super::{
    DeltaNetChunkedCausalInvMetalKernel, DeltaNetChunkedCumsumMetalKernel, DeltaNetChunkedGramAMetalKernel,
    DeltaNetChunkedOutputAndStateMetalKernel, DeltaNetPrefillPrepMetalKernel,
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
const FORCE_CHUNKED_ENV: &str = "UZU_FORCE_GDN_CHUNKED_PREFILL";
const DISABLE_CHUNKED_ENV: &str = "UZU_DISABLE_GDN_CHUNKED_PREFILL";

pub struct MetalDeltaNetChunkedPrefill {
    min_t: usize,
    prep: DeltaNetPrefillPrepMetalKernel,
    cumsum: DeltaNetChunkedCumsumMetalKernel,
    gram_a: DeltaNetChunkedGramAMetalKernel,
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
            prep: DeltaNetPrefillPrepMetalKernel::new(context, outer_data_type, head_dim, true)?,
            cumsum: DeltaNetChunkedCumsumMetalKernel::new(context, CHUNK_SIZE as u32)?,
            gram_a: DeltaNetChunkedGramAMetalKernel::new(context, head_dim, CHUNK_SIZE as u32)?,
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
        if std::env::var_os(DISABLE_CHUNKED_ENV).is_some() {
            return false;
        }
        std::env::var_os(FORCE_CHUNKED_ENV).is_some() || suffix_len >= self.min_t
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
        self.gram_a.encode(
            &q_norm,
            &k_norm,
            &g,
            &beta,
            &mut qk,
            &mut a_packed,
            &mut a_inv,
            args.num_heads,
            args.num_groups,
            args.key_dim,
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
#[path = "../../../../../../unit/backends/metal/kernel/gdn/chunked_test.rs"]
mod tests;
