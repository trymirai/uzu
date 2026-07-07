use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, AllocationType, Backend, Context, Encoder, Kernels,
        kernel::{
            Conv1dPackKernel, DeltaNetChunkedCumsumKernel, DeltaNetChunkedGramKernel, DeltaNetChunkedMegaApplyKernel,
            DeltaNetChunkedPrepKernel, DeltaNetChunkedSolveKernel, DeltaNetChunkedSolveTKernel, DeltaNetConvScanKernel,
            DeltaNetConvUpdateKernel, DeltaNetNormGateKernel, DeltaNetPrefillKernel, DeltaNetPrefillPrepKernel,
            DeltaNetUpdateKernel,
        },
    },
    config::token_mixer::delta_net::DeltaNetConfig,
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::{Linear, LinearBlockError},
        mixer::{Mixer, MixerState, attention::rope::PrecalculatedRoPE},
    },
    parameters::{ParameterLoaderError, ParameterTree},
    utils::maybe_mut::MaybeMut,
};

/// Minimum prefill length (tokens) for chunked Mode-L with the MXU backend.
pub const CHUNKED_MXU_MIN_T: usize = 256;

const CHUNKED_CHUNK_SIZE: usize = 64;
const CHUNKED_BLOCK_SIZE: usize = 16;
const CHUNKED_VT: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnPrefillPath {
    Recurrent,
    ChunkedModeL,
}

pub fn select_gdn_prefill_path<C: Context>(
    context: &C,
    suffix_len: usize,
) -> GdnPrefillPath {
    route_gdn_prefill(context.supports_mxu(), suffix_len)
}

fn route_gdn_prefill(
    supports_mxu: bool,
    suffix_len: usize,
) -> GdnPrefillPath {
    if supports_mxu && suffix_len >= CHUNKED_MXU_MIN_T {
        GdnPrefillPath::ChunkedModeL
    } else {
        GdnPrefillPath::Recurrent
    }
}

pub struct DeltaNetState<B: Backend> {
    conv_state: Allocation<B>,
    ssm_state: Allocation<B>,
    suffix_length: Option<usize>,
}

impl<B: Backend> MixerState<B> for DeltaNetState<B> {
    fn prepare(
        &mut self,
        _context_length: usize,
        _suffix_length: usize,
        _context: &B::Context,
    ) -> Result<(), B::Error> {
        Ok(())
    }

    fn encode_accept(
        &mut self,
        accepted_indices: &[usize],
        _encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        assert!(self.suffix_length.take() == Some(*accepted_indices.last().unwrap() + 1));
        Ok(())
    }
}

pub struct DeltaNet<B: Backend> {
    num_heads: usize,
    head_dim: usize,
    num_groups: usize,
    value_head_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    total_proj_dim: usize,
    kernel_size: usize,
    outer_data_type: DataType,
    in_projection: Box<dyn Linear<B>>,
    conv_weight: Allocation<B>,
    conv_bias: Option<Allocation<B>>,
    conv_update: <B::Kernels as Kernels>::DeltaNetConvUpdateKernel,
    conv_pack: <B::Kernels as Kernels>::Conv1dPackKernel,
    conv_scan: <B::Kernels as Kernels>::DeltaNetConvScanKernel,
    a_log: Allocation<B>,
    dt_bias: Allocation<B>,
    norm_weight: Allocation<B>,
    norm_epsilon: f32,
    delta_net_update: <B::Kernels as Kernels>::DeltaNetUpdateKernel,
    delta_net_prefill_prep: <B::Kernels as Kernels>::DeltaNetPrefillPrepKernel,
    delta_net_prefill: <B::Kernels as Kernels>::DeltaNetPrefillKernel,
    delta_net_norm_gate: <B::Kernels as Kernels>::DeltaNetNormGateKernel,
    chunked_prep: Option<<B::Kernels as Kernels>::DeltaNetChunkedPrepKernel>,
    chunked_cumsum: Option<<B::Kernels as Kernels>::DeltaNetChunkedCumsumKernel>,
    chunked_gram: Option<<B::Kernels as Kernels>::DeltaNetChunkedGramKernel>,
    chunked_solve: Option<<B::Kernels as Kernels>::DeltaNetChunkedSolveKernel>,
    chunked_solve_t: Option<<B::Kernels as Kernels>::DeltaNetChunkedSolveTKernel>,
    chunked_mega: Option<<B::Kernels as Kernels>::DeltaNetChunkedMegaApplyKernel>,
    out_projection: Box<dyn Linear<B>>,
}

#[derive(Debug, Error)]
pub enum DeltaNetNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Unsupported configuration: {0}")]
    UnsupportedConfiguration(String),
}

impl<B: Backend> DeltaNet<B> {
    pub fn new(
        hidden_dim: usize,
        outer_data_type: DataType,
        config: &DeltaNetConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<(Self, Option<Allocation<B>>), DeltaNetNewError<B>> {
        if config.kernel_size < 2 {
            return Err(DeltaNetNewError::UnsupportedConfiguration(format!(
                "kernel_size must be >= 2, got {}",
                config.kernel_size
            )));
        }
        if config.head_dim != 128 {
            return Err(DeltaNetNewError::UnsupportedConfiguration(format!(
                "head_dim must be 128, got {}",
                config.head_dim
            )));
        }
        if config.value_head_dim != 128 {
            return Err(DeltaNetNewError::UnsupportedConfiguration(format!(
                "value_head_dim must be 128, got {}",
                config.value_head_dim
            )));
        }

        let inner_data_type = DataType::F32;

        let key_dim = config.num_groups * config.head_dim;
        let value_dim = config.num_heads * config.value_head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let total_proj_dim = conv_dim + value_dim + config.num_heads + config.num_heads;

        let (in_projection, in_projection_input_hadamard_factors) =
            <dyn Linear<B>>::new_extracting_input_hadamard_mixed_precision(
                hidden_dim,
                [total_proj_dim],
                false,
                context,
                outer_data_type,
                outer_data_type,
                outer_data_type,
                &parameter_tree.subtree("in_proj")?,
            )?;

        let conv_config = &config.conv_config;
        let conv_tree = parameter_tree.subtree("conv")?;

        let conv_weight =
            conv_tree.leaf("weights")?.validate(&[conv_dim, config.kernel_size], inner_data_type)?.read_allocation()?;
        let conv_bias = if conv_config.has_biases {
            Some(conv_tree.leaf("biases")?.validate(&[conv_dim], inner_data_type)?.read_allocation()?)
        } else {
            None
        };
        let conv_update =
            <B::Kernels as Kernels>::DeltaNetConvUpdateKernel::new(context, outer_data_type, conv_config.has_biases)
                .map_err(DeltaNetNewError::Backend)?;
        let conv_pack = <B::Kernels as Kernels>::Conv1dPackKernel::new(context, inner_data_type, outer_data_type)
            .map_err(DeltaNetNewError::Backend)?;
        let conv_scan =
            <B::Kernels as Kernels>::DeltaNetConvScanKernel::new(context, outer_data_type, conv_config.has_biases)
                .map_err(DeltaNetNewError::Backend)?;

        let a_log = parameter_tree.leaf("a_log")?.validate(&[config.num_heads], inner_data_type)?.read_allocation()?;
        let dt_bias =
            parameter_tree.leaf("dt_bias")?.validate(&[config.num_heads], inner_data_type)?.read_allocation()?;
        let norm_weight = parameter_tree
            .leaf("norm.scales")?
            .validate(&[config.value_head_dim], inner_data_type)?
            .read_allocation()?;
        let norm_epsilon = config.norm_config.epsilon;
        let delta_net_update =
            <B::Kernels as Kernels>::DeltaNetUpdateKernel::new(context, outer_data_type, config.head_dim as u32)
                .map_err(DeltaNetNewError::Backend)?;
        let delta_net_prefill_prep =
            <B::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(context, outer_data_type, config.head_dim as u32)
                .map_err(DeltaNetNewError::Backend)?;
        let delta_net_prefill =
            <B::Kernels as Kernels>::DeltaNetPrefillKernel::new(context, outer_data_type, config.head_dim as u32)
                .map_err(DeltaNetNewError::Backend)?;
        let delta_net_norm_gate = <B::Kernels as Kernels>::DeltaNetNormGateKernel::new(context, outer_data_type)
            .map_err(DeltaNetNewError::Backend)?;

        let (chunked_prep, chunked_cumsum, chunked_gram, chunked_solve, chunked_solve_t, chunked_mega) =
            if context.supports_mxu() {
                let prep = <B::Kernels as Kernels>::DeltaNetChunkedPrepKernel::new(
                    context,
                    outer_data_type,
                    config.head_dim as u32,
                )
                .map_err(DeltaNetNewError::Backend)?;
                let cumsum = <B::Kernels as Kernels>::DeltaNetChunkedCumsumKernel::new(context)
                    .map_err(DeltaNetNewError::Backend)?;
                let gram = <B::Kernels as Kernels>::DeltaNetChunkedGramKernel::new(
                    context,
                    config.head_dim as u32,
                    CHUNKED_CHUNK_SIZE as u32,
                )
                .map_err(DeltaNetNewError::Backend)?;
                let solve =
                    <B::Kernels as Kernels>::DeltaNetChunkedSolveKernel::new(context, CHUNKED_CHUNK_SIZE as u32, false)
                        .map_err(DeltaNetNewError::Backend)?;
                let solve_t = <B::Kernels as Kernels>::DeltaNetChunkedSolveTKernel::new(
                    context,
                    CHUNKED_CHUNK_SIZE as u32,
                    CHUNKED_VT as u32,
                )
                .map_err(DeltaNetNewError::Backend)?;
                let mega = <B::Kernels as Kernels>::DeltaNetChunkedMegaApplyKernel::new(
                    context,
                    outer_data_type,
                    outer_data_type,
                    CHUNKED_VT as u32,
                    true,
                )
                .map_err(DeltaNetNewError::Backend)?;
                (Some(prep), Some(cumsum), Some(gram), Some(solve), Some(solve_t), Some(mega))
            } else {
                (None, None, None, None, None, None)
            };

        let out_projection = <dyn Linear<B>>::new_mixed_precision(
            value_dim,
            [hidden_dim],
            false,
            context,
            outer_data_type,
            outer_data_type,
            outer_data_type,
            &parameter_tree.subtree("out_proj")?,
        )?;

        Ok((
            Self {
                num_heads: config.num_heads,
                head_dim: config.head_dim,
                num_groups: config.num_groups,
                value_head_dim: config.value_head_dim,
                key_dim,
                value_dim,
                conv_dim,
                total_proj_dim,
                kernel_size: config.kernel_size,
                outer_data_type,
                in_projection,
                conv_weight,
                conv_bias,
                conv_update,
                conv_pack,
                conv_scan,
                a_log,
                dt_bias,
                norm_weight,
                norm_epsilon,
                delta_net_update,
                delta_net_prefill_prep,
                delta_net_prefill,
                delta_net_norm_gate,
                chunked_prep,
                chunked_cumsum,
                chunked_gram,
                chunked_solve,
                chunked_solve_t,
                chunked_mega,
                out_projection,
            },
            in_projection_input_hadamard_factors,
        ))
    }

    fn encode_recurrent_prefill(
        &self,
        in_projected: &Allocation<B>,
        state: &mut DeltaNetState<B>,
        suffix_len: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut delta_output =
            encoder.allocate_scratch(size_for_shape(&[suffix_len, self.value_dim], self.outer_data_type))?;
        let mut prep_q_norm = encoder.allocate_scratch(size_for_shape(&[suffix_len, self.key_dim], DataType::F32))?;
        let mut prep_k_norm = encoder.allocate_scratch(size_for_shape(&[suffix_len, self.key_dim], DataType::F32))?;
        let mut prep_beta = encoder.allocate_scratch(size_for_shape(&[suffix_len, self.num_heads], DataType::F32))?;
        let mut prep_decay = encoder.allocate_scratch(size_for_shape(&[suffix_len, self.num_heads], DataType::F32))?;
        self.delta_net_prefill_prep.encode(
            in_projected,
            &self.a_log,
            &self.dt_bias,
            &mut prep_q_norm,
            &mut prep_k_norm,
            &mut prep_beta,
            &mut prep_decay,
            self.num_heads as u32,
            self.num_groups as u32,
            self.key_dim as u32,
            self.value_dim as u32,
            suffix_len as u32,
            encoder,
        );
        self.delta_net_prefill.encode(
            &prep_q_norm,
            &prep_k_norm,
            &prep_beta,
            &prep_decay,
            in_projected,
            &mut state.ssm_state,
            &mut delta_output,
            self.num_heads as u32,
            self.num_groups as u32,
            self.value_head_dim as u32,
            self.key_dim as u32,
            self.value_dim as u32,
            suffix_len as u32,
            self.value_head_dim.div_ceil(16) as u32,
            encoder,
        );
        self.delta_net_norm_gate.encode(
            &mut delta_output,
            in_projected,
            &self.norm_weight,
            self.num_heads as u32,
            self.value_head_dim as u32,
            self.value_dim as u32,
            self.conv_dim as u32,
            self.total_proj_dim as u32,
            self.norm_epsilon,
            suffix_len as u32,
            encoder,
        );
        Ok(delta_output)
    }

    pub(crate) fn encode_chunked_prefill(
        &self,
        in_projected: &Allocation<B>,
        state: &mut DeltaNetState<B>,
        suffix_len: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let prep = self.chunked_prep.as_ref().expect("chunked prep kernel present when routed to chunked path");
        let cumsum = self.chunked_cumsum.as_ref().expect("chunked cumsum kernel present when routed to chunked path");
        let gram = self.chunked_gram.as_ref().expect("chunked gram kernel present when routed to chunked path");
        let solve = self.chunked_solve.as_ref().expect("chunked solve kernel present when routed to chunked path");
        let solve_t = self.chunked_solve_t.as_ref().expect("chunked solveT kernel present when routed to chunked path");
        let mega = self.chunked_mega.as_ref().expect("chunked mega kernel present when routed to chunked path");

        let chunk_size = CHUNKED_CHUNK_SIZE;
        let block_size = CHUNKED_BLOCK_SIZE;
        let num_chunks = suffix_len.div_ceil(chunk_size);
        let num_blocks = chunk_size.div_ceil(block_size);
        let num_col_pairs = num_blocks.div_ceil(2);

        let mut q_norm = encoder.allocate_scratch(size_for_shape(&[suffix_len * self.key_dim], DataType::F32))?;
        let mut k_norm = encoder.allocate_scratch(size_for_shape(&[suffix_len * self.key_dim], DataType::F32))?;
        let mut beta = encoder.allocate_scratch(size_for_shape(&[suffix_len * self.num_heads], DataType::F32))?;
        let mut log_decay = encoder.allocate_scratch(size_for_shape(&[suffix_len * self.num_heads], DataType::F32))?;
        let mut g = encoder.allocate_scratch(size_for_shape(&[suffix_len * self.num_heads], DataType::F32))?;
        let mut kk = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * self.num_groups * chunk_size * chunk_size],
            DataType::F32,
        ))?;
        let mut qk = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * self.num_heads * chunk_size * chunk_size],
            DataType::F32,
        ))?;
        let mut a_packed = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * self.num_heads * num_blocks * num_col_pairs * block_size * 2 * block_size],
            DataType::F32,
        ))?;
        let mut a_inv = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * self.num_heads * num_blocks * block_size * block_size],
            DataType::F32,
        ))?;
        let mut t_mat = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * self.num_heads * chunk_size * chunk_size],
            DataType::BF16,
        ))?;
        let mut delta_output =
            encoder.allocate_scratch(size_for_shape(&[suffix_len, self.value_dim], self.outer_data_type))?;

        prep.encode(
            in_projected,
            &self.a_log,
            &self.dt_bias,
            &mut q_norm,
            &mut k_norm,
            &mut beta,
            &mut log_decay,
            self.num_heads as u32,
            self.num_groups as u32,
            self.key_dim as u32,
            self.value_dim as u32,
            suffix_len as u32,
            encoder,
        );
        cumsum.encode(&log_decay, &mut g, self.num_heads as u32, suffix_len as u32, chunk_size as u32, encoder);
        gram.encode(
            &q_norm,
            &k_norm,
            &g,
            &mut kk,
            &mut qk,
            self.num_heads as u32,
            self.num_groups as u32,
            self.key_dim as u32,
            suffix_len as u32,
            encoder,
        );
        solve.encode(
            &kk,
            &beta,
            &g,
            &mut a_packed,
            &mut a_inv,
            self.num_heads as u32,
            self.num_groups as u32,
            suffix_len as u32,
            encoder,
        );
        solve_t.encode(&a_packed, &a_inv, &mut t_mat, self.num_heads as u32, suffix_len as u32, encoder);
        mega.encode(
            &q_norm,
            &k_norm,
            in_projected,
            &qk,
            &t_mat,
            &g,
            &beta,
            &mut state.ssm_state,
            &mut delta_output,
            self.num_heads as u32,
            self.num_groups as u32,
            self.value_head_dim as u32,
            self.key_dim as u32,
            self.value_dim as u32,
            suffix_len as u32,
            encoder,
        );
        self.delta_net_norm_gate.encode(
            &mut delta_output,
            in_projected,
            &self.norm_weight,
            self.num_heads as u32,
            self.value_head_dim as u32,
            self.value_dim as u32,
            self.conv_dim as u32,
            self.total_proj_dim as u32,
            self.norm_epsilon,
            suffix_len as u32,
            encoder,
        );
        Ok(delta_output)
    }
}

impl<B: Backend> Mixer<B> for DeltaNet<B> {
    fn speculation_supported(&self) -> bool {
        false
    }

    fn max_context_length(&self) -> Option<usize> {
        None
    }

    fn create_empty_state(
        &self,
        _max_context_length: Option<usize>,
        context: &B::Context,
    ) -> Result<Box<dyn MixerState<B>>, B::Error> {
        let mut conv_state = context.create_allocation(
            size_for_shape(&[self.conv_dim, self.kernel_size - 1], DataType::F32),
            AllocationType::Global,
        )?;

        let mut ssm_state = context.create_allocation(
            size_for_shape(&[self.num_heads, self.value_head_dim, self.head_dim], DataType::F32),
            AllocationType::Global,
        )?;

        let mut zero_encoder = Encoder::<B>::new(context)?;
        zero_encoder.encode_fill(&mut conv_state, 0);
        zero_encoder.encode_fill(&mut ssm_state, 0);
        zero_encoder.end_encoding().submit().wait_until_completed()?;

        Ok(Box::new(DeltaNetState {
            conv_state,
            ssm_state,
            suffix_length: None,
        }))
    }

    fn encode(
        &self,
        hidden: Allocation<B>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        assert!(precalculated_rope.is_none(), "unexpected rope for delta net mixer");

        if !batch_dim.full_accept() {
            panic!("delta net doesn't support speculation");
        }

        let state = state.expect("delta net requires state");
        let state = state.downcast::<DeltaNetState<B>>().expect("incorrect type of delta net state");
        let MaybeMut::Mut(state) = state else {
            panic!("delta net doesn't support immutable state");
        };

        assert!(state.suffix_length.is_none(), "delta net called with state with unaccepted tokens");

        let mut in_projected = self.in_projection.encode(hidden, batch_dim.size(), encoder)?;

        let delta_output = if batch_dim.size() == 1 {
            let mut delta_output =
                encoder.allocate_scratch(size_for_shape(&[batch_dim.size(), self.value_dim], self.outer_data_type))?;
            self.conv_update.encode(
                &self.conv_weight,
                self.conv_bias.as_ref(),
                &mut in_projected,
                &mut state.conv_state,
                self.kernel_size as u32,
                self.conv_dim as u32,
                (self.kernel_size - 1) as u32,
                encoder,
            );

            self.delta_net_update.encode(
                &in_projected,
                &self.a_log,
                &self.dt_bias,
                &self.norm_weight,
                &mut state.ssm_state,
                &mut delta_output,
                self.num_heads as u32,
                self.num_groups as u32,
                self.value_head_dim as u32,
                self.key_dim as u32,
                self.value_dim as u32,
                self.norm_epsilon,
                encoder,
            );
            delta_output
        } else {
            let mut padded = encoder.allocate_scratch(size_for_shape(
                &[batch_dim.size() + self.kernel_size - 1, self.total_proj_dim],
                DataType::F32,
            ))?;
            self.conv_pack.encode(
                &state.conv_state,
                &in_projected,
                &mut padded,
                (self.kernel_size - 1) as u32,
                self.total_proj_dim as u32,
                batch_dim.size() as u32,
                self.conv_dim as u32,
                encoder,
            );
            self.conv_scan.encode(
                &padded,
                &self.conv_weight,
                self.conv_bias.as_ref(),
                &mut in_projected,
                &mut state.conv_state,
                batch_dim.size() as u32,
                self.kernel_size as u32,
                self.total_proj_dim as u32,
                (self.kernel_size - 1) as u32,
                self.conv_dim as u32,
                self.total_proj_dim as u32,
                encoder,
            );
            match select_gdn_prefill_path(encoder.context(), batch_dim.size()) {
                GdnPrefillPath::ChunkedModeL => {
                    self.encode_chunked_prefill(&in_projected, state, batch_dim.size(), encoder)?
                },
                GdnPrefillPath::Recurrent => {
                    self.encode_recurrent_prefill(&in_projected, state, batch_dim.size(), encoder)?
                },
            }
        };

        state.suffix_length = Some(batch_dim.size());

        self.out_projection.encode(delta_output, batch_dim.size(), encoder)
    }
}

#[cfg(test)]
mod router_tests {
    use proc_macros::uzu_test;

    use super::{CHUNKED_MXU_MIN_T, GdnPrefillPath, route_gdn_prefill};

    #[uzu_test]
    fn mxu_routes_chunked_above_threshold() {
        assert_eq!(route_gdn_prefill(true, CHUNKED_MXU_MIN_T - 1), GdnPrefillPath::Recurrent);
        assert_eq!(route_gdn_prefill(true, CHUNKED_MXU_MIN_T), GdnPrefillPath::ChunkedModeL);
    }

    #[uzu_test]
    fn non_mxu_always_recurrent() {
        for suffix_len in [1, CHUNKED_MXU_MIN_T, 1024, usize::MAX] {
            assert_eq!(route_gdn_prefill(false, suffix_len), GdnPrefillPath::Recurrent);
        }
    }
}
