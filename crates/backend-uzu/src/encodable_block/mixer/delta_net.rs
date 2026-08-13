use std::mem::size_of;

use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, AllocationType, Backend, Context, Encoder, Kernels,
        gpu_types::trie::TrieNode,
        kernel::{
            Conv1dPackKernel, ConvTreeScanKernel, DeltaNetConvScanKernel, DeltaNetConvUpdateKernel,
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
            Mixer, MixerState,
            attention::rope::PrecalculatedRoPE,
            delta_net::tree_verify::{TreeVerifyEncodeArguments, TreeVerifyNewArguments},
        },
    },
    parameters::{ParameterLoaderError, ParameterTree},
    utils::maybe_mut::MaybeMut,
};

pub(crate) mod tree_verify;

const INNER_DATA_TYPE: DataType = DataType::F32;

enum DeltaNetSuffixStatus<B: Backend> {
    Flat {
        suffix_length: u32,
    },
    Tree {
        conv_states: Allocation<B>,
        k: Allocation<B>,
        v: Allocation<B>,
        log_decay: Allocation<B>,
        beta: Allocation<B>,
        parents: Box<[i32]>,
    },
}

pub struct DeltaNetState<B: Backend> {
    conv_state: Allocation<B>,
    ssm_state: Allocation<B>,
    suffix_status: Option<DeltaNetSuffixStatus<B>>,
    state_advance: <B::Kernels as Kernels>::StateAdvanceKernel,
}

impl<B: Backend> MixerState<B> for DeltaNetState<B> {
    fn prepare(
        &mut self,
        _context_length: u32,
        _suffix_length: u32,
        _context: &B::Context,
    ) -> Result<(), B::Error> {
        Ok(())
    }

    fn encode_accept(
        &mut self,
        accepted_indices: &[u32],
        encoder: &mut Encoder<B>,
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
                encoder.encode_copy(
                    &conv_states,
                    accepted_offset..accepted_offset + conv_state_size,
                    &mut self.conv_state,
                    ..,
                );

                let mut accepted_indices_buffer =
                    encoder.allocate_constant(size_for_shape(&[accepted_indices.len() as u32], DataType::U32))?;
                accepted_indices_buffer.copyin(accepted_indices);
                self.state_advance.encode(
                    &k,
                    &v,
                    &log_decay,
                    &beta,
                    &accepted_indices_buffer,
                    &mut self.ssm_state,
                    accepted_indices.len() as u32,
                    encoder,
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
    conv_weight: Allocation<B>,
    conv_bias: Option<Allocation<B>>,
    conv_update: <B::Kernels as Kernels>::DeltaNetConvUpdateKernel,
    conv_pack: <B::Kernels as Kernels>::Conv1dPackKernel,
    conv_scan: <B::Kernels as Kernels>::DeltaNetConvScanKernel,
    conv_tree_scan: <B::Kernels as Kernels>::ConvTreeScanKernel,
    a_log: Allocation<B>,
    dt_bias: Allocation<B>,
    norm_weight: Allocation<B>,
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
                &parameter_tree.subtree("in_proj"),
            )?;

        let conv_config = &config.conv_config;
        let conv_tree = parameter_tree.subtree("conv");

        let conv_weight =
            conv_tree.leaf("weights")?.validate(&[conv_dim, config.kernel_size], INNER_DATA_TYPE)?.read_allocation()?;
        let conv_bias = if conv_config.has_biases {
            Some(conv_tree.leaf("biases")?.validate(&[conv_dim], INNER_DATA_TYPE)?.read_allocation()?)
        } else {
            None
        };
        let conv_update =
            <B::Kernels as Kernels>::DeltaNetConvUpdateKernel::new(context, outer_data_type, conv_config.has_biases)
                .map_err(DeltaNetNewError::Backend)?;
        let conv_pack = <B::Kernels as Kernels>::Conv1dPackKernel::new(context, INNER_DATA_TYPE, outer_data_type)
            .map_err(DeltaNetNewError::Backend)?;
        let conv_scan =
            <B::Kernels as Kernels>::DeltaNetConvScanKernel::new(context, outer_data_type, conv_config.has_biases)
                .map_err(DeltaNetNewError::Backend)?;
        let conv_tree_scan = <B::Kernels as Kernels>::ConvTreeScanKernel::new(
            context,
            outer_data_type,
            config.kernel_size,
            conv_config.has_biases,
        )
        .map_err(DeltaNetNewError::Backend)?;

        let a_log = parameter_tree.leaf("a_log")?.validate(&[config.num_heads], INNER_DATA_TYPE)?.read_allocation()?;
        let dt_bias =
            parameter_tree.leaf("dt_bias")?.validate(&[config.num_heads], INNER_DATA_TYPE)?.read_allocation()?;
        let norm_weight = parameter_tree
            .leaf("norm.scales")?
            .validate(&[config.value_head_dim], INNER_DATA_TYPE)?
            .read_allocation()?;
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
            if <<B::Kernels as Kernels>::DeltaNetTreeVerify as DeltaNetTreeVerifyTrait<B>>::is_supported(context) {
                Some(
                    <<B::Kernels as Kernels>::DeltaNetTreeVerify as DeltaNetTreeVerifyTrait<B>>::new(
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

        let chunked = <B::Kernels as Kernels>::DeltaNetChunkedPrefill::new(context, outer_data_type, config.head_dim)
            .map_err(DeltaNetNewError::Backend)?;

        let out_projection = <dyn Linear<B>>::new_mixed_precision(
            value_dim,
            [hidden_dim],
            false,
            context,
            outer_data_type,
            outer_data_type,
            outer_data_type,
            &parameter_tree.subtree("out_proj"),
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
        in_projected: Allocation<B>,
        batch_dim: &BatchTopology,
        state: &mut DeltaNetState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let tree_verify = self.tree_verify.as_ref().expect("DeltaNet tree verification is unsupported");
        let tree_size = batch_dim.size();
        let mut parents = encoder.allocate_constant(size_for_shape(&[tree_size], DataType::I32))?;
        parents.copyin(batch_dim.parents());
        let mut trie = encoder.allocate_constant(tree_size as usize * size_of::<TrieNode>())?;
        trie.copyin(batch_dim.nodes());

        let mut conv_states =
            encoder.allocate_scratch_for_shape(&[tree_size, self.conv_dim, self.kernel_size - 1], INNER_DATA_TYPE)?;
        let mut k = encoder.allocate_scratch_for_shape(&[tree_size, self.key_dim], self.outer_data_type)?;
        let mut v = encoder.allocate_scratch_for_shape(&[tree_size, self.value_dim], self.outer_data_type)?;
        let mut beta = encoder.allocate_scratch_for_shape(&[tree_size, self.num_heads], INNER_DATA_TYPE)?;
        let mut log_decay = encoder.allocate_scratch_for_shape(&[tree_size, self.num_heads], INNER_DATA_TYPE)?;

        let mut tree_projected =
            encoder.allocate_scratch_for_shape(&[tree_size, self.total_proj_dim], self.outer_data_type)?;
        let mut q = encoder.allocate_scratch_for_shape(&[tree_size, self.key_dim], self.outer_data_type)?;

        self.conv_tree_scan.encode(
            &in_projected,
            &self.conv_weight,
            self.conv_bias.as_ref(),
            &state.conv_state,
            &parents,
            &mut tree_projected,
            &mut conv_states,
            tree_size,
            self.total_proj_dim,
            self.conv_dim,
            encoder,
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
            encoder,
        );

        let mut delta_output = tree_verify.encode(
            TreeVerifyEncodeArguments {
                q: &q,
                k: &k,
                v: &v,
                trie: &trie,
                log_decay: &log_decay,
                beta: &beta,
                h0: &state.ssm_state,
                tree_size,
            },
            encoder,
        )?;
        self.delta_net_norm_gate.encode(
            &mut delta_output,
            &tree_projected,
            &self.norm_weight,
            self.num_heads,
            self.value_head_dim,
            self.value_dim,
            self.conv_dim,
            self.total_proj_dim,
            self.norm_epsilon,
            tree_size,
            encoder,
        );

        let output = self.out_projection.encode(delta_output, tree_size, encoder)?;
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

    fn max_context_length(&self) -> Option<usize> {
        None
    }

    fn create_empty_state(
        &self,
        _max_context_length: Option<usize>,
        context: &B::Context,
    ) -> Result<Box<dyn MixerState<B>>, B::Error> {
        let mut conv_state = context.create_allocation(
            size_for_shape(&[self.conv_dim, self.kernel_size - 1], INNER_DATA_TYPE),
            AllocationType::Global,
        )?;

        let mut ssm_state = context.create_allocation(
            size_for_shape(&[self.num_heads, self.value_head_dim, self.head_dim], INNER_DATA_TYPE),
            AllocationType::Global,
        )?;

        let mut zero_encoder = Encoder::<B>::new(context)?;
        zero_encoder.encode_fill(&mut conv_state, 0);
        zero_encoder.encode_fill(&mut ssm_state, 0);
        zero_encoder.end_encoding().submit().wait_until_completed()?;

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
        encoder.push_debug_group("delta net");

        assert!(precalculated_rope.is_none(), "unexpected rope for delta net mixer");

        let state = state.expect("delta net requires state");
        let state = state.downcast::<DeltaNetState<B>>().expect("incorrect type of delta net state");
        let MaybeMut::Mut(state) = state else {
            panic!("delta net doesn't support immutable state");
        };

        assert!(state.suffix_status.is_none(), "delta net called with state with an unaccepted suffix");

        let mut in_projected = self.in_projection.encode(hidden, batch_dim.size(), encoder)?;

        if !batch_dim.full_accept() {
            let output = self.encode_tree_verify(in_projected, batch_dim, state, encoder)?;

            encoder.pop_debug_group();

            return Ok(output);
        }

        let mut delta_output =
            encoder.allocate_scratch_for_shape(&[batch_dim.size(), self.value_dim], self.outer_data_type)?;
        if batch_dim.size() == 1 {
            self.conv_update.encode(
                &self.conv_weight,
                self.conv_bias.as_ref(),
                &mut in_projected,
                &mut state.conv_state,
                self.kernel_size,
                self.conv_dim,
                self.kernel_size - 1,
                encoder,
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
                encoder,
            );
        } else {
            let mut padded = encoder.allocate_scratch_for_shape(
                &[batch_dim.size() + self.kernel_size - 1, self.total_proj_dim],
                INNER_DATA_TYPE,
            )?;
            self.conv_pack.encode(
                &state.conv_state,
                &in_projected,
                &mut padded,
                self.kernel_size - 1,
                self.total_proj_dim,
                batch_dim.size(),
                self.conv_dim,
                encoder,
            );
            self.conv_scan.encode(
                &padded,
                &self.conv_weight,
                self.conv_bias.as_ref(),
                &mut in_projected,
                &mut state.conv_state,
                batch_dim.size(),
                self.kernel_size,
                self.total_proj_dim,
                self.kernel_size - 1,
                self.conv_dim,
                self.total_proj_dim,
                encoder,
            );
            if let Some(chunked) = self.chunked.as_ref().filter(|chunked| chunked.should_use(batch_dim.size())) {
                chunked.encode(
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
                    encoder,
                )?;
            } else {
                let mut prep_q_norm =
                    encoder.allocate_scratch_for_shape(&[batch_dim.size(), self.key_dim], INNER_DATA_TYPE)?;
                let mut prep_k_norm =
                    encoder.allocate_scratch_for_shape(&[batch_dim.size(), self.key_dim], INNER_DATA_TYPE)?;
                let mut prep_beta =
                    encoder.allocate_scratch_for_shape(&[batch_dim.size(), self.num_heads], INNER_DATA_TYPE)?;
                let mut prep_decay =
                    encoder.allocate_scratch_for_shape(&[batch_dim.size(), self.num_heads], INNER_DATA_TYPE)?;
                self.delta_net_prefill_prep.encode(
                    &in_projected,
                    &self.a_log,
                    &self.dt_bias,
                    &mut prep_q_norm,
                    &mut prep_k_norm,
                    None::<&mut Allocation<B>>,
                    &mut prep_beta,
                    &mut prep_decay,
                    self.num_heads,
                    self.num_groups,
                    self.key_dim,
                    self.value_dim,
                    batch_dim.size(),
                    encoder,
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
                    encoder,
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
                encoder,
            );
        }

        state.suffix_status = Some(DeltaNetSuffixStatus::Flat {
            suffix_length: batch_dim.size(),
        });

        let output = self.out_projection.encode(delta_output, batch_dim.size(), encoder)?;

        encoder.pop_debug_group();

        Ok(output)
    }
}
