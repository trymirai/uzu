use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, Buffer, BufferRef, CommandBuffer, CommandBufferEncoding, Context, Kernels,
        kernel::{
            ConvTreeScanKernel, DeltaNetConvPackKernel, DeltaNetConvScanKernel, DeltaNetConvUpdateKernel,
            DeltaNetNormGateKernel, DeltaNetPrefillKernel, DeltaNetPrefillPrepKernel, DeltaNetUpdateKernel,
            StateAdvanceKernel,
            delta_net_chunked_prefill::{DeltaNetChunkedPrefill, DeltaNetChunkedPrefillArgs},
            delta_net_tree_verify::DeltaNetTreeVerify as DeltaNetTreeVerifyTrait,
        },
    },
    config::token_mixer::delta_net::DeltaNetConfig,
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::{Linear, LinearBlockError},
        mixer::{
            Mixer, MixerEncodeError, MixerState,
            attention::rope::PrecalculatedRoPE,
            delta_net::tree_verify::{TreeVerifyEncodeArguments, TreeVerifyNewArguments},
        },
    },
    safetensors::{ParameterLoaderError, ParameterTree},
    utils::{maybe_mut::MaybeMut, size_for_shape::size_for_shape},
};

pub(crate) mod tree_verify;

const INNER_DATA_TYPE: DataType = DataType::F32;
// Prefill conv: up to CONV_TILES token tiles of at least CONV_MIN_TILE_TOKENS (correctness needs >= KERNEL_SIZE - 1).
const CONV_TILES: u32 = 16;
const CONV_MIN_TILE_TOKENS: u32 = 16;

enum DeltaNetSuffixStatus<B: Backend> {
    Flat {
        suffix_length: u32,
    },
    Tree {
        conv_states: B::ScratchBuffer,
        k: B::ScratchBuffer,
        v: B::ScratchBuffer,
        log_decay: B::ScratchBuffer,
        beta: B::ScratchBuffer,
        parents: Box<[i32]>,
    },
}

pub struct DeltaNetState<B: Backend> {
    conv_state: B::GlobalBuffer,
    ssm_state: B::GlobalBuffer,
    suffix_status: Option<DeltaNetSuffixStatus<B>>,
    state_advance: <B::Kernels as Kernels>::StateAdvanceKernel,
    is_empty: bool,
}

impl<B: Backend> MixerState<B> for DeltaNetState<B> {
    fn prepare(
        &mut self,
        _context_length: u32,
        _suffix_length: u32,
    ) -> Result<(), B::Error> {
        Ok(())
    }

    fn set_context_length(
        &mut self,
        _context_length: u32,
    ) {
    }

    fn encode_accept(
        &mut self,
        accepted_indices: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let suffix_status = self.suffix_status.take().expect("delta net state has no suffix to accept");
        let accepted_index = *accepted_indices.last().expect("delta net state attempted to accept zero indices");

        match suffix_status {
            DeltaNetSuffixStatus::Flat {
                suffix_length,
            } => {
                assert!(accepted_index == suffix_length - 1, "attempted a partial flat delta net accept");
            },
            DeltaNetSuffixStatus::Tree {
                conv_states,
                k,
                v,
                log_decay,
                beta,
                parents,
            } => {
                assert!(accepted_indices.iter().all(|&index| (index as usize) < parents.len()));
                assert_eq!(parents[accepted_indices[0] as usize], -1);
                assert!(accepted_indices.windows(2).all(|edge| parents[edge[1] as usize] == edge[0] as i32));

                let conv_state_size = self.conv_state.size();
                let accepted_offset = accepted_index as usize * conv_state_size;
                command_buffer.encode_copy(
                    conv_states.subrange(accepted_offset..accepted_offset + conv_state_size),
                    &mut self.conv_state,
                );

                let accepted_indices_buffer = command_buffer.allocate_constant_from_slice(accepted_indices)?;
                self.state_advance.encode(
                    &k,
                    &v,
                    &log_decay,
                    &beta,
                    &accepted_indices_buffer,
                    &mut self.ssm_state,
                    accepted_indices.len() as u32,
                    command_buffer,
                );
            },
        }
        Ok(())
    }
}

pub struct DeltaNet<B: Backend> {
    num_heads: u32,
    head_dim: u32,
    num_groups: u32,
    value_head_dim: u32,
    key_dim: u32,
    value_dim: u32,
    conv_dim: u32,
    total_proj_dim: u32,
    kernel_size: u32,
    outer_data_type: DataType,
    in_projection: Box<dyn Linear<B>>,
    conv_weight: B::GlobalBuffer,
    conv_bias: Option<B::GlobalBuffer>,
    conv_update: <B::Kernels as Kernels>::DeltaNetConvUpdateKernel,
    conv_pack: <B::Kernels as Kernels>::DeltaNetConvPackKernel,
    conv_scan: <B::Kernels as Kernels>::DeltaNetConvScanKernel,
    conv_tree_scan: <B::Kernels as Kernels>::ConvTreeScanKernel,
    a_log: B::GlobalBuffer,
    dt_bias: B::GlobalBuffer,
    norm_weight: B::GlobalBuffer,
    norm_epsilon: f32,
    delta_net_update: <B::Kernels as Kernels>::DeltaNetUpdateKernel,
    delta_net_prefill_prep: <B::Kernels as Kernels>::DeltaNetPrefillPrepKernel,
    delta_net_tree_prep: <B::Kernels as Kernels>::DeltaNetPrefillPrepKernel,
    delta_net_prefill: <B::Kernels as Kernels>::DeltaNetPrefillKernel,
    delta_net_norm_gate: <B::Kernels as Kernels>::DeltaNetNormGateKernel,
    tree_verify: Option<<B::Kernels as Kernels>::DeltaNetTreeVerify>,
    chunked: Option<<B::Kernels as Kernels>::DeltaNetChunkedPrefill>,
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
        hidden_dim: u32,
        outer_data_type: DataType,
        config: &DeltaNetConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<(Self, Option<B::GlobalBuffer>), DeltaNetNewError<B>> {
        if config.kernel_size != 4 {
            return Err(DeltaNetNewError::UnsupportedConfiguration(format!(
                "kernel_size must be 4, got {}",
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

        let key_dim = config.num_groups * config.head_dim;
        let value_dim = config.num_heads * config.value_head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let total_proj_dim = conv_dim + value_dim + config.num_heads + config.num_heads;
        // The prefill conv scan moves 8 channels per thread as vec4 pairs: the in_proj row stride must be a multiple of 8.
        if !total_proj_dim.is_multiple_of(8) {
            return Err(DeltaNetNewError::UnsupportedConfiguration(format!(
                "in_proj width must be a multiple of 8, got {total_proj_dim}"
            )));
        }

        let (in_projection, in_projection_input_hadamard_factors) =
            <dyn Linear<B>>::new_with_input_rht_mixed_precision(
                hidden_dim,
                total_proj_dim,
                outer_data_type,
                outer_data_type,
                outer_data_type,
                false,
                &parameter_tree.subtree("in_proj"),
                context,
            )?;

        let conv_config = &config.conv_config;
        let conv_tree = parameter_tree.subtree("conv");

        let conv_weight = conv_tree.read_buffer("weights", &[conv_dim, config.kernel_size], INNER_DATA_TYPE)?;
        let conv_bias = if conv_config.has_biases {
            Some(conv_tree.read_buffer("biases", &[conv_dim], INNER_DATA_TYPE)?)
        } else {
            None
        };
        let conv_update =
            <B::Kernels as Kernels>::DeltaNetConvUpdateKernel::new(context, outer_data_type, conv_config.has_biases)
                .map_err(DeltaNetNewError::Backend)?;
        let conv_pack =
            <B::Kernels as Kernels>::DeltaNetConvPackKernel::new(context, outer_data_type, config.kernel_size)
                .map_err(DeltaNetNewError::Backend)?;
        let conv_scan = <B::Kernels as Kernels>::DeltaNetConvScanKernel::new(
            context,
            outer_data_type,
            config.kernel_size,
            conv_config.has_biases,
        )
        .map_err(DeltaNetNewError::Backend)?;
        let conv_tree_scan = <B::Kernels as Kernels>::ConvTreeScanKernel::new(
            context,
            outer_data_type,
            config.kernel_size,
            conv_config.has_biases,
        )
        .map_err(DeltaNetNewError::Backend)?;

        let a_log = parameter_tree.read_buffer("a_log", &[config.num_heads], INNER_DATA_TYPE)?;
        let dt_bias = parameter_tree.read_buffer("dt_bias", &[config.num_heads], INNER_DATA_TYPE)?;
        let norm_weight = parameter_tree.read_buffer("norm.scales", &[config.value_head_dim], INNER_DATA_TYPE)?;
        let norm_epsilon = config.norm_config.epsilon;
        let delta_net_update =
            <B::Kernels as Kernels>::DeltaNetUpdateKernel::new(context, outer_data_type, config.head_dim)
                .map_err(DeltaNetNewError::Backend)?;
        let delta_net_prefill_prep = <B::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
            context,
            outer_data_type,
            INNER_DATA_TYPE,
            config.head_dim,
            false,
            false,
        )
        .map_err(DeltaNetNewError::Backend)?;
        let delta_net_tree_prep = <B::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
            context,
            outer_data_type,
            outer_data_type,
            config.head_dim,
            true,
            true,
        )
        .map_err(DeltaNetNewError::Backend)?;
        let delta_net_prefill =
            <B::Kernels as Kernels>::DeltaNetPrefillKernel::new(context, outer_data_type, config.head_dim)
                .map_err(DeltaNetNewError::Backend)?;
        let delta_net_norm_gate = <B::Kernels as Kernels>::DeltaNetNormGateKernel::new(context, outer_data_type)
            .map_err(DeltaNetNewError::Backend)?;
        let tree_verify =
            if <<B::Kernels as Kernels>::DeltaNetTreeVerify as DeltaNetTreeVerifyTrait>::is_supported(context) {
                Some(
                    <<B::Kernels as Kernels>::DeltaNetTreeVerify as DeltaNetTreeVerifyTrait>::new(
                        context,
                        &TreeVerifyNewArguments {
                            data_type: outer_data_type,
                            num_k_heads: config.num_groups,
                            num_v_heads: config.num_heads,
                            head_k_dim: config.head_dim,
                            head_v_dim: config.value_head_dim,
                        },
                    )
                    .map_err(DeltaNetNewError::Backend)?,
                )
            } else {
                None
            };

        let chunked = <B::Kernels as Kernels>::DeltaNetChunkedPrefill::new(
            context,
            outer_data_type,
            config.head_dim,
            config.num_heads / config.num_groups,
        )
        .map_err(DeltaNetNewError::Backend)?;

        let out_projection = <dyn Linear<B>>::new_mixed_precision(
            value_dim,
            hidden_dim,
            outer_data_type,
            outer_data_type,
            outer_data_type,
            false,
            &parameter_tree.subtree("out_proj"),
            context,
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
                conv_tree_scan,
                a_log,
                dt_bias,
                norm_weight,
                norm_epsilon,
                delta_net_update,
                delta_net_prefill_prep,
                delta_net_tree_prep,
                delta_net_prefill,
                delta_net_norm_gate,
                tree_verify,
                chunked,
                out_projection,
            },
            in_projection_input_hadamard_factors,
        ))
    }

    fn encode_tree_verify(
        &self,
        in_projected: impl BufferRef<Backend = B>,
        batch_dim: &BatchTopology,
        state: &mut DeltaNetState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, MixerEncodeError<B>> {
        let tree_verify = self.tree_verify.as_ref().expect("DeltaNet tree verification is unsupported");
        let tree_size = batch_dim.size();
        let parents =
            command_buffer.allocate_constant_from_slice(batch_dim.parents()).map_err(MixerEncodeError::Backend)?;
        let trie = command_buffer.allocate_constant_from_slice(batch_dim.nodes()).map_err(MixerEncodeError::Backend)?;

        let mut conv_states = command_buffer
            .allocate_scratch_for_shape(&[tree_size, self.conv_dim, self.kernel_size - 1], INNER_DATA_TYPE)
            .map_err(MixerEncodeError::Backend)?;
        let mut k = command_buffer
            .allocate_scratch_for_shape(&[tree_size, self.key_dim], self.outer_data_type)
            .map_err(MixerEncodeError::Backend)?;
        let mut v = command_buffer
            .allocate_scratch_for_shape(&[tree_size, self.value_dim], self.outer_data_type)
            .map_err(MixerEncodeError::Backend)?;
        let mut beta = command_buffer
            .allocate_scratch_for_shape(&[tree_size, self.num_heads], INNER_DATA_TYPE)
            .map_err(MixerEncodeError::Backend)?;
        let mut log_decay = command_buffer
            .allocate_scratch_for_shape(&[tree_size, self.num_heads], INNER_DATA_TYPE)
            .map_err(MixerEncodeError::Backend)?;

        let mut tree_projected = command_buffer
            .allocate_scratch_for_shape(&[tree_size, self.total_proj_dim], self.outer_data_type)
            .map_err(MixerEncodeError::Backend)?;
        let mut q = command_buffer
            .allocate_scratch_for_shape(&[tree_size, self.key_dim], self.outer_data_type)
            .map_err(MixerEncodeError::Backend)?;

        self.conv_tree_scan.encode(
            in_projected,
            &self.conv_weight,
            self.conv_bias.as_ref(),
            &state.conv_state,
            &parents,
            &mut tree_projected,
            &mut conv_states,
            tree_size,
            self.total_proj_dim,
            self.conv_dim,
            command_buffer,
        );

        self.delta_net_tree_prep.encode(
            &tree_projected,
            &self.a_log,
            &self.dt_bias,
            &mut q,
            &mut k,
            Some(&mut v),
            &mut beta,
            &mut log_decay,
            self.num_heads,
            self.num_groups,
            self.key_dim,
            self.value_dim,
            tree_size,
            command_buffer,
        );

        let delta_output = tree_verify
            .encode(
                TreeVerifyEncodeArguments {
                    q: &q,
                    k: &k,
                    v: &v,
                    trie: &trie,
                    log_decay: &log_decay,
                    beta: &beta,
                    h0: &state.ssm_state,
                    in_proj: &tree_projected,
                    norm_weight: &self.norm_weight,
                    conv_dim: self.conv_dim,
                    total_proj_dim: self.total_proj_dim,
                    norm_epsilon: self.norm_epsilon,
                    tree_size,
                },
                command_buffer,
            )
            .map_err(MixerEncodeError::Backend)?;

        let output = self.out_projection.encode(delta_output, tree_size, command_buffer)?;
        state.suffix_status = Some(DeltaNetSuffixStatus::Tree {
            conv_states,
            k,
            v,
            log_decay,
            beta,
            parents: batch_dim.parents().into(),
        });
        Ok(output)
    }
}

impl<B: Backend> Mixer<B> for DeltaNet<B> {
    fn speculation_supported(&self) -> bool {
        self.tree_verify.is_some()
    }

    fn max_context_length(&self) -> Option<u32> {
        None
    }

    fn create_empty_state(
        &self,
        _max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<Box<dyn MixerState<B>>, B::Error> {
        let conv_state =
            context.create_buffer(size_for_shape(&[self.conv_dim, self.kernel_size - 1], INNER_DATA_TYPE))?;
        let ssm_state = context
            .create_buffer(size_for_shape(&[self.num_heads, self.value_head_dim, self.head_dim], INNER_DATA_TYPE))?;

        let state_advance = <B::Kernels as Kernels>::StateAdvanceKernel::new(
            context,
            self.outer_data_type,
            self.head_dim,
            self.num_heads,
            self.num_groups,
        )?;

        Ok(Box::new(DeltaNetState {
            conv_state,
            ssm_state,
            suffix_status: None,
            state_advance,
            is_empty: true,
        }))
    }

    fn encode(
        &self,
        hidden: B::ScratchBuffer,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, MixerEncodeError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("delta net");

        assert!(precalculated_rope.is_none(), "unexpected rope for delta net mixer");

        let state = state.expect("delta net requires state");
        let state = state.downcast::<DeltaNetState<B>>().expect("incorrect type of delta net state");
        let MaybeMut::Mut(state) = state else {
            panic!("delta net doesn't support immutable state");
        };

        assert!(state.suffix_status.is_none(), "delta net called with state with an unaccepted suffix");

        if state.is_empty {
            command_buffer.encode_fill(&mut state.conv_state, 0);
            command_buffer.encode_fill(&mut state.ssm_state, 0);
            state.is_empty = false;
        }

        let mut in_projected = self.in_projection.encode(hidden, batch_dim.size(), command_buffer)?;

        if !batch_dim.full_accept() {
            let output = self.encode_tree_verify(&in_projected, batch_dim, state, command_buffer)?;

            return Ok(output);
        }

        let mut delta_output = command_buffer
            .allocate_scratch_for_shape(&[batch_dim.size(), self.value_dim], self.outer_data_type)
            .map_err(MixerEncodeError::Backend)?;
        if batch_dim.size() == 1 {
            self.conv_update.encode(
                &self.conv_weight,
                self.conv_bias.as_ref(),
                &mut in_projected,
                &mut state.conv_state,
                self.kernel_size,
                self.conv_dim,
                self.kernel_size - 1,
                command_buffer,
            );

            self.delta_net_update.encode(
                &in_projected,
                &self.a_log,
                &self.dt_bias,
                &self.norm_weight,
                &mut state.ssm_state,
                &mut delta_output,
                self.num_heads,
                self.num_groups,
                self.value_head_dim,
                self.key_dim,
                self.value_dim,
                self.norm_epsilon,
                command_buffer,
            );
        } else {
            let conv_tile_tokens = batch_dim.size().div_ceil(CONV_TILES).max(CONV_MIN_TILE_TOKENS);
            debug_assert!(conv_tile_tokens >= self.kernel_size - 1);
            let mut packed = command_buffer
                .allocate_scratch_for_shape(&[CONV_TILES, self.kernel_size - 1, self.conv_dim], INNER_DATA_TYPE)
                .map_err(MixerEncodeError::Backend)?;
            self.conv_pack.encode(
                &state.conv_state,
                &in_projected,
                &mut packed,
                batch_dim.size(),
                conv_tile_tokens,
                self.total_proj_dim,
                self.conv_dim,
                command_buffer,
            );
            self.conv_scan.encode(
                &packed,
                &self.conv_weight,
                self.conv_bias.as_ref(),
                &mut in_projected,
                &mut state.conv_state,
                batch_dim.size(),
                conv_tile_tokens,
                self.conv_dim,
                self.total_proj_dim,
                command_buffer,
            );
            if let Some(chunked) = self.chunked.as_ref().filter(|chunked| chunked.should_use(batch_dim.size())) {
                chunked
                    .encode(
                        DeltaNetChunkedPrefillArgs {
                            in_projected: &in_projected,
                            a_log: &self.a_log,
                            dt_bias: &self.dt_bias,
                            ssm_state: &mut state.ssm_state,
                            delta_output: &mut delta_output,
                            num_heads: self.num_heads,
                            num_groups: self.num_groups,
                            value_head_dim: self.value_head_dim,
                            key_dim: self.key_dim,
                            value_dim: self.value_dim,
                            suffix_len: batch_dim.size(),
                        },
                        command_buffer,
                    )
                    .map_err(MixerEncodeError::Backend)?;
            } else {
                let mut prep_q_norm = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim.size(), self.key_dim], INNER_DATA_TYPE)
                    .map_err(MixerEncodeError::Backend)?;
                let mut prep_k_norm = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim.size(), self.key_dim], INNER_DATA_TYPE)
                    .map_err(MixerEncodeError::Backend)?;
                let mut prep_beta = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim.size(), self.num_heads], INNER_DATA_TYPE)
                    .map_err(MixerEncodeError::Backend)?;
                let mut prep_decay = command_buffer
                    .allocate_scratch_for_shape(&[batch_dim.size(), self.num_heads], INNER_DATA_TYPE)
                    .map_err(MixerEncodeError::Backend)?;
                self.delta_net_prefill_prep.encode(
                    &in_projected,
                    &self.a_log,
                    &self.dt_bias,
                    &mut prep_q_norm,
                    &mut prep_k_norm,
                    None::<&mut B::ScratchBuffer>,
                    &mut prep_beta,
                    &mut prep_decay,
                    self.num_heads,
                    self.num_groups,
                    self.key_dim,
                    self.value_dim,
                    batch_dim.size(),
                    command_buffer,
                );
                self.delta_net_prefill.encode(
                    &prep_q_norm,
                    &prep_k_norm,
                    &prep_beta,
                    &prep_decay,
                    &in_projected,
                    &mut state.ssm_state,
                    &mut delta_output,
                    self.num_heads,
                    self.num_groups,
                    self.value_head_dim,
                    self.key_dim,
                    self.value_dim,
                    batch_dim.size(),
                    self.value_head_dim.div_ceil(16),
                    command_buffer,
                );
            }
            self.delta_net_norm_gate.encode(
                &mut delta_output,
                &in_projected,
                &self.norm_weight,
                self.num_heads,
                self.value_head_dim,
                self.value_dim,
                self.conv_dim,
                self.total_proj_dim,
                self.norm_epsilon,
                batch_dim.size(),
                command_buffer,
            );
        }

        state.suffix_status = Some(DeltaNetSuffixStatus::Flat {
            suffix_length: batch_dim.size(),
        });

        let output = self.out_projection.encode(delta_output, batch_dim.size(), command_buffer)?;

        Ok(output)
    }
}
