use std::ops::{Deref, DerefMut};

use thiserror::Error;

use crate::{
    DataType,
    backends::common::{
        Backend, Encoder, Kernels,
        kernel::{
            Conv1dPackKernel, DeltaNetConvScanKernel, DeltaNetConvUpdateKernel, DeltaNetNormGateKernel,
            DeltaNetPrefillKernel, DeltaNetPrefillPrepKernel, DeltaNetUpdateKernel,
        },
    },
    config::DeltaNetAttentionConfig,
    encodable_block::linear::{Linear, LinearBlockError},
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree, resolve_subtree},
};

#[derive(Debug, Error)]
pub enum DeltaNetMixerError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Unsupported configuration: {0}")]
    UnsupportedConfiguration(String),
    #[error("Linear error: {0}")]
    InnerLinearError(#[from] Box<LinearBlockError<B>>),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
}

pub(crate) struct DeltaNetMixer<B: Backend> {
    layer_index: usize,
    config: DeltaNetAttentionConfig,
    in_projection: Box<dyn Linear<B>>,
    out_projection: Box<dyn Linear<B>>,
    // Decode kernels
    conv_update: <B::Kernels as Kernels>::DeltaNetConvUpdateKernel,
    delta_net_update: <B::Kernels as Kernels>::DeltaNetUpdateKernel,
    // Prefill kernels
    conv_pack: <B::Kernels as Kernels>::Conv1dPackKernel,
    conv_scan: <B::Kernels as Kernels>::DeltaNetConvScanKernel,
    prefill_prep: <B::Kernels as Kernels>::DeltaNetPrefillPrepKernel,
    delta_net_prefill: <B::Kernels as Kernels>::DeltaNetPrefillKernel,
    norm_gate: <B::Kernels as Kernels>::DeltaNetNormGateKernel,
    // Parameters
    conv_weight: B::DenseBuffer,
    conv_bias: Option<B::DenseBuffer>,
    a_log: B::DenseBuffer,
    dt_bias: B::DenseBuffer,
    norm_weight: B::DenseBuffer,
}

impl<B: Backend> DeltaNetMixer<B> {
    pub(crate) fn new(
        context: &B::Context,
        config: DeltaNetAttentionConfig,
        layer_index: usize,
        model_dim: usize,
        decoder_layer_loader: &ParameterTree<B::Context>,
    ) -> Result<Self, DeltaNetMixerError<B>> {
        if config.kernel_size < 2 {
            return Err(DeltaNetMixerError::UnsupportedConfiguration(format!(
                "kernel_size must be >= 2, got {}",
                config.kernel_size
            )));
        }
        if config.head_dim != 128 {
            return Err(DeltaNetMixerError::UnsupportedConfiguration(format!(
                "head_dim must be 128, got {}",
                config.head_dim
            )));
        }
        if config.value_head_dim != 128 {
            return Err(DeltaNetMixerError::UnsupportedConfiguration(format!(
                "value_head_dim must be 128, got {}",
                config.value_head_dim
            )));
        }

        let data_type: DataType = config.in_proj_config.activation_precision().into();
        let has_bias = config.conv_config.has_biases;

        // Load weights
        let mixer_tree = resolve_subtree(decoder_layer_loader, &["mixer"]);
        let conv_tree = resolve_subtree(&mixer_tree, &["conv", "conv1d"]);

        let in_projection = <dyn Linear<B>>::new(
            &config.in_proj_config,
            false,
            model_dim,
            [config.total_proj_dim()],
            context,
            &resolve_subtree(decoder_layer_loader, &["mixer.in_projection", "mixer.in_proj"]),
            ArrayId::Main,
            ArrayId::SsmInProj,
        )
        .map_err(|e| DeltaNetMixerError::InnerLinearError(Box::new(e)))?;

        let out_projection = <dyn Linear<B>>::new(
            &config.out_proj_config,
            false,
            config.value_dim(),
            [model_dim],
            context,
            &resolve_subtree(decoder_layer_loader, &["mixer.out_projection", "mixer.out_proj"]),
            ArrayId::AttentionOutput,
            ArrayId::Main,
        )
        .map_err(|e| DeltaNetMixerError::InnerLinearError(Box::new(e)))?;

        let conv_weight = conv_tree.leaf("weights")?.read_buffer()?;
        let conv_bias = if has_bias {
            Some(conv_tree.leaf("biases")?.read_buffer()?)
        } else {
            None
        };

        let a_log = mixer_tree.leaf("a_log")?.read_buffer()?;
        let dt_bias = mixer_tree.leaf("dt_bias")?.read_buffer()?;
        let norm_tree = resolve_subtree(&mixer_tree, &["norm", "inner_norm"]);
        let norm_weight = norm_tree.leaf("scales")?.read_buffer()?;

        // Create kernels
        let conv_update = <B::Kernels as Kernels>::DeltaNetConvUpdateKernel::new(context, data_type, has_bias)
            .map_err(DeltaNetMixerError::BackendError)?;
        let delta_net_update =
            <B::Kernels as Kernels>::DeltaNetUpdateKernel::new(context, data_type, config.head_dim as u32)
                .map_err(DeltaNetMixerError::BackendError)?;
        let conv_pack = <B::Kernels as Kernels>::Conv1dPackKernel::new(context, data_type)
            .map_err(DeltaNetMixerError::BackendError)?;
        let conv_scan = <B::Kernels as Kernels>::DeltaNetConvScanKernel::new(context, data_type, has_bias)
            .map_err(DeltaNetMixerError::BackendError)?;
        let prefill_prep =
            <B::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(context, data_type, config.head_dim as u32)
                .map_err(DeltaNetMixerError::BackendError)?;
        let delta_net_prefill =
            <B::Kernels as Kernels>::DeltaNetPrefillKernel::new(context, data_type, config.head_dim as u32)
                .map_err(DeltaNetMixerError::BackendError)?;
        let norm_gate = <B::Kernels as Kernels>::DeltaNetNormGateKernel::new(context, data_type)
            .map_err(DeltaNetMixerError::BackendError)?;

        Ok(Self {
            layer_index,
            config,
            in_projection,
            out_projection,
            conv_update,
            delta_net_update,
            conv_pack,
            conv_scan,
            prefill_prep,
            delta_net_prefill,
            norm_gate,
            conv_weight,
            conv_bias,
            a_log,
            dt_bias,
            norm_weight,
        })
    }

    fn run_conv_update(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) {
        let in_proj = state.array(ArrayId::SsmInProj);
        let conv_state = state.array(ArrayId::DeltaNetConvState(self.layer_index));

        let kernel_size = self.config.kernel_size;

        self.conv_update.encode(
            &self.conv_weight,
            self.conv_bias.as_ref(),
            in_proj.buffer().borrow_mut().deref_mut(),
            conv_state.buffer().borrow_mut().deref_mut(),
            kernel_size as u32,
            self.config.conv_dim() as u32,
            (kernel_size - 1) as u32,
            encoder,
        );
    }

    fn run_conv_scan(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
        suffix_length: usize,
    ) {
        let kernel_size = self.config.kernel_size;
        let state_stride = kernel_size - 1;
        let conv_dim = self.config.conv_dim();
        let total_proj_dim = self.config.total_proj_dim();

        let in_proj = state.array(ArrayId::SsmInProj);
        let conv_state = state.array(ArrayId::DeltaNetConvState(self.layer_index));

        let padded = state.conv_padded_buffer().expect("Missing conv padded buffer");

        self.conv_pack.encode(
            conv_state.buffer().borrow().deref(),
            in_proj.buffer().borrow().deref(),
            padded.buffer().borrow_mut().deref_mut(),
            state_stride as u32,
            total_proj_dim as u32,
            suffix_length as u32,
            conv_dim as u32,
            encoder,
        );

        self.conv_scan.encode(
            padded.buffer().borrow().deref(),
            &self.conv_weight,
            self.conv_bias.as_ref(),
            in_proj.buffer().borrow_mut().deref_mut(),
            conv_state.buffer().borrow_mut().deref_mut(),
            suffix_length as u32,
            kernel_size as u32,
            total_proj_dim as u32,
            state_stride as u32,
            conv_dim as u32,
            total_proj_dim as u32,
            encoder,
        );
    }

    fn run_delta_rule(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) {
        let in_proj = state.array(ArrayId::SsmInProj);
        let ssm_state = state.array(ArrayId::DeltaNetSsmState(self.layer_index));
        let out = state.array(ArrayId::AttentionOutput);

        self.delta_net_update.encode(
            in_proj.buffer().borrow().deref(),
            &self.a_log,
            &self.dt_bias,
            &self.norm_weight,
            ssm_state.buffer().borrow_mut().deref_mut(),
            out.buffer().borrow_mut().deref_mut(),
            self.config.num_heads as u32,
            self.config.num_groups as u32,
            self.config.value_head_dim as u32,
            self.config.key_dim() as u32,
            self.config.value_dim() as u32,
            self.config.norm_config.epsilon,
            encoder,
        );
    }

    fn run_delta_rule_prefill(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
        suffix_length: usize,
    ) {
        let num_dv_groups = ((self.config.value_head_dim + 7) / 8) as u32;

        let in_proj = state.array(ArrayId::SsmInProj);
        let ssm_state = state.array(ArrayId::DeltaNetSsmState(self.layer_index));
        let out = state.array(ArrayId::AttentionOutput);

        let aux = state.llm_aux.as_ref().expect("DeltaNet prep buffers not initialized");
        let prep_q_norm = aux.delta_net_prep_q_norm.as_ref().expect("DeltaNet prep_q_norm not initialized");
        let prep_k_norm = aux.delta_net_prep_k_norm.as_ref().expect("DeltaNet prep_k_norm not initialized");
        let prep_beta = aux.delta_net_prep_beta.as_ref().expect("DeltaNet prep_beta not initialized");
        let prep_decay = aux.delta_net_prep_decay.as_ref().expect("DeltaNet prep_decay not initialized");

        self.prefill_prep.encode(
            in_proj.buffer().borrow().deref(),
            &self.a_log,
            &self.dt_bias,
            prep_q_norm.buffer().borrow_mut().deref_mut(),
            prep_k_norm.buffer().borrow_mut().deref_mut(),
            prep_beta.buffer().borrow_mut().deref_mut(),
            prep_decay.buffer().borrow_mut().deref_mut(),
            self.config.num_heads as u32,
            self.config.num_groups as u32,
            self.config.key_dim() as u32,
            self.config.value_dim() as u32,
            suffix_length as u32,
            encoder,
        );

        self.delta_net_prefill.encode(
            prep_q_norm.buffer().borrow().deref(),
            prep_k_norm.buffer().borrow().deref(),
            prep_beta.buffer().borrow().deref(),
            prep_decay.buffer().borrow().deref(),
            in_proj.buffer().borrow().deref(),
            ssm_state.buffer().borrow_mut().deref_mut(),
            out.buffer().borrow_mut().deref_mut(),
            self.config.num_heads as u32,
            self.config.num_groups as u32,
            self.config.value_head_dim as u32,
            self.config.key_dim() as u32,
            self.config.value_dim() as u32,
            suffix_length as u32,
            num_dv_groups,
            encoder,
        );

        // Norm gate
        self.norm_gate.encode(
            out.buffer().borrow_mut().deref_mut(),
            in_proj.buffer().borrow().deref(),
            &self.norm_weight,
            self.config.num_heads as u32,
            self.config.value_head_dim as u32,
            self.config.value_dim() as u32,
            self.config.conv_dim() as u32,
            self.config.total_proj_dim() as u32,
            self.config.norm_config.epsilon,
            suffix_length as u32,
            encoder,
        );
    }

    pub(crate) fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let active_row_count = state.active_row_count();
        if active_row_count == 0 {
            return Ok(());
        }

        self.in_projection.encode(state, encoder)?;

        if active_row_count == 1 {
            self.run_conv_update(state, encoder);
            self.run_delta_rule(state, encoder);
        } else {
            self.run_conv_scan(state, encoder, active_row_count);
            self.run_delta_rule_prefill(state, encoder, active_row_count);
        }

        self.out_projection.encode(state, encoder)?;
        Ok(())
    }
}
