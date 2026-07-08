use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, AllocationType, Backend, Context, Encoder, Kernels,
        kernel::{
            Conv1dPackKernel, DeltaNetConvScanKernel, DeltaNetConvUpdateKernel, DeltaNetNormGateKernel,
            DeltaNetPrefillKernel, DeltaNetPrefillPrepKernel, DeltaNetUpdateKernel,
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

pub mod chunked_prefill;

use chunked_prefill::{DeltaNetChunkedPrefill, DeltaNetChunkedPrefillArgs};

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
        let delta_net_prefill_prep = <B::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
            context,
            outer_data_type,
            config.head_dim as u32,
            false,
        )
        .map_err(DeltaNetNewError::Backend)?;
        let delta_net_prefill =
            <B::Kernels as Kernels>::DeltaNetPrefillKernel::new(context, outer_data_type, config.head_dim as u32)
                .map_err(DeltaNetNewError::Backend)?;
        let delta_net_norm_gate = <B::Kernels as Kernels>::DeltaNetNormGateKernel::new(context, outer_data_type)
            .map_err(DeltaNetNewError::Backend)?;

        let chunked =
            <B::Kernels as Kernels>::DeltaNetChunkedPrefill::new(context, outer_data_type, config.head_dim as u32)
                .map_err(DeltaNetNewError::Backend)?;

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
                chunked,
                out_projection,
            },
            in_projection_input_hadamard_factors,
        ))
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

        let mut delta_output =
            encoder.allocate_scratch(size_for_shape(&[batch_dim.size(), self.value_dim], self.outer_data_type))?;
        if batch_dim.size() == 1 {
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
            if let Some(chunked) = self.chunked.as_ref().filter(|chunked| chunked.should_use(batch_dim.size())) {
                chunked.encode(
                    DeltaNetChunkedPrefillArgs {
                        in_projected: &in_projected,
                        a_log: &self.a_log,
                        dt_bias: &self.dt_bias,
                        ssm_state: &mut state.ssm_state,
                        delta_output: &mut delta_output,
                        num_heads: self.num_heads as u32,
                        num_groups: self.num_groups as u32,
                        value_head_dim: self.value_head_dim as u32,
                        key_dim: self.key_dim as u32,
                        value_dim: self.value_dim as u32,
                        suffix_len: batch_dim.size(),
                    },
                    encoder,
                )?;
            } else {
                let mut prep_q_norm =
                    encoder.allocate_scratch(size_for_shape(&[batch_dim.size(), self.key_dim], DataType::F32))?;
                let mut prep_k_norm =
                    encoder.allocate_scratch(size_for_shape(&[batch_dim.size(), self.key_dim], DataType::F32))?;
                let mut prep_beta =
                    encoder.allocate_scratch(size_for_shape(&[batch_dim.size(), self.num_heads], DataType::F32))?;
                let mut prep_decay =
                    encoder.allocate_scratch(size_for_shape(&[batch_dim.size(), self.num_heads], DataType::F32))?;
                self.delta_net_prefill_prep.encode(
                    &in_projected,
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
                    batch_dim.size() as u32,
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
                    self.num_heads as u32,
                    self.num_groups as u32,
                    self.value_head_dim as u32,
                    self.key_dim as u32,
                    self.value_dim as u32,
                    batch_dim.size() as u32,
                    self.value_head_dim.div_ceil(16) as u32,
                    encoder,
                );
            }
            self.delta_net_norm_gate.encode(
                &mut delta_output,
                &in_projected,
                &self.norm_weight,
                self.num_heads as u32,
                self.value_head_dim as u32,
                self.value_dim as u32,
                self.conv_dim as u32,
                self.total_proj_dim as u32,
                self.norm_epsilon,
                batch_dim.size() as u32,
                encoder,
            );
        }

        state.suffix_length = Some(batch_dim.size());

        self.out_projection.encode(delta_output, batch_dim.size(), encoder)
    }
}
