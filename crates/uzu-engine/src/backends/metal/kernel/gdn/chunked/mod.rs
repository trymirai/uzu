use super::super::{
    DeltaNetChunkedOutputAndStateMetalKernel, DeltaNetChunkedPrecomputeMetalKernel, DeltaNetPrefillPrepMetalKernel,
};
use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding,
            kernel::{
                DeltaNetPrefillPrepKernel,
                delta_net_chunked_prefill::{CHUNK_SIZE, DeltaNetChunkedPrefill, DeltaNetChunkedPrefillArgs},
            },
        },
        metal::{Metal, command_buffer::MetalCommandBufferEncoding, context::MetalContext},
    },
    data_type::DataType,
};

const MXU_MIN_T: u32 = 256;
const VT: u32 = 32;
const INNER_DATA_TYPE: DataType = DataType::F32;

pub struct MetalDeltaNetChunkedPrefill {
    min_t: u32,
    prep: DeltaNetPrefillPrepMetalKernel,
    precompute: DeltaNetChunkedPrecomputeMetalKernel,
    output_and_state: DeltaNetChunkedOutputAndStateMetalKernel,
}

impl DeltaNetChunkedPrefill for MetalDeltaNetChunkedPrefill {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        outer_data_type: DataType,
        head_dim: u32,
        groups_per_head: u32,
    ) -> Result<Option<Self>, <Metal as Backend>::Error> {
        if outer_data_type == DataType::F16 {
            return Ok(None);
        }

        if !context.supports_mxu {
            return Ok(None);
        }

        // The precompute kernel assigns one grouped v-head to each of its four simdgroups.
        if groups_per_head > 4 {
            return Ok(None);
        }

        let scratch_data_type = if outer_data_type == DataType::BF16 {
            DataType::BF16
        } else {
            INNER_DATA_TYPE
        };

        Ok(Some(Self {
            min_t: MXU_MIN_T,
            prep: DeltaNetPrefillPrepMetalKernel::new(
                context,
                outer_data_type,
                INNER_DATA_TYPE,
                head_dim,
                true,
                false,
            )?,
            precompute: DeltaNetChunkedPrecomputeMetalKernel::new(context, head_dim, CHUNK_SIZE)?,
            output_and_state: DeltaNetChunkedOutputAndStateMetalKernel::new(
                context,
                outer_data_type,
                outer_data_type,
                scratch_data_type,
                head_dim,
                VT,
                true,
            )?,
        }))
    }

    fn should_use(
        &self,
        suffix_len: u32,
    ) -> bool {
        suffix_len >= self.min_t
    }

    fn encode(
        &self,
        args: DeltaNetChunkedPrefillArgs<'_, Metal>,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(), <Metal as Backend>::Error> {
        let suffix_len = args.suffix_len;
        let num_chunks = suffix_len.div_ceil(CHUNK_SIZE);

        let mut q_norm = command_buffer.allocate_scratch_for_shape(&[suffix_len, args.key_dim], INNER_DATA_TYPE)?;
        let mut k_norm = command_buffer.allocate_scratch_for_shape(&[suffix_len, args.key_dim], INNER_DATA_TYPE)?;
        let mut beta = command_buffer.allocate_scratch_for_shape(&[suffix_len, args.num_heads], INNER_DATA_TYPE)?;
        let mut log_decay =
            command_buffer.allocate_scratch_for_shape(&[suffix_len, args.num_heads], INNER_DATA_TYPE)?;
        let mut g = command_buffer.allocate_scratch_for_shape(&[suffix_len, args.num_heads], INNER_DATA_TYPE)?;
        let mut qk = command_buffer
            .allocate_scratch_for_shape(&[num_chunks, args.num_heads, CHUNK_SIZE, CHUNK_SIZE], INNER_DATA_TYPE)?;
        let mut t_mat = command_buffer
            .allocate_scratch_for_shape(&[num_chunks, args.num_heads, CHUNK_SIZE, CHUNK_SIZE], DataType::BF16)?;

        self.prep.encode(
            args.in_projected,
            args.a_log,
            args.dt_bias,
            &mut q_norm,
            &mut k_norm,
            None::<&mut <Metal as Backend>::ScratchBuffer>,
            &mut beta,
            &mut log_decay,
            args.num_heads,
            args.num_groups,
            args.key_dim,
            args.value_dim,
            suffix_len,
            command_buffer,
        );
        self.precompute.encode(
            &q_norm,
            &k_norm,
            &log_decay,
            &beta,
            &mut g,
            &mut qk,
            &mut t_mat,
            args.num_heads,
            args.num_groups,
            args.key_dim,
            suffix_len,
            command_buffer,
        );
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
            suffix_len,
            command_buffer,
        );
        Ok(())
    }
}
