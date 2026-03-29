use std::ops::{Deref, DerefMut};

use crate::{
    DataType,
    array::Array,
    backends::common::{
        Backend, Encoder, Kernels,
        kernel::{
            Conv1dPackKernel, DeltaNetConvScanKernel, DeltaNetConvUpdateKernel, DeltaNetNormGateKernel,
            DeltaNetPrefillKernel, DeltaNetPrefillPrepKernel, DeltaNetUpdateKernel,
        },
    },
    config::{DecoderLayerType, DeltaNetAttentionConfig},
    encodable_block::linear::Linear,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterTree, resolve_subtree},
};

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
    conv_weight: Array<B>,
    conv_bias: Option<Array<B>>,
    a_log: Array<B>,
    dt_bias: Array<B>,
    norm_weight: Array<B>,
}

impl<B: Backend> DeltaNetMixer<B> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        context: &B::Context,
        layer_type: DecoderLayerType,
        config: DeltaNetAttentionConfig,
        layer_index: usize,
        model_dim: usize,
        decoder_layer_loader: &ParameterTree<B::Context>,
    ) -> Self {
        if !matches!(layer_type, DecoderLayerType::DeltaNet { .. }) {
            panic!("Layer {} marked as non-DeltaNet but DeltaNet config provided", layer_index);
        }
        assert!(config.kernel_size >= 2, "DeltaNet requires kernel_size >= 2, got {}", config.kernel_size);
        if config.head_dim > 128 {
            todo!("DeltaNet prefill kernel supports head_k_dim <= 128, got {}", config.head_dim);
        }
        if config.value_head_dim > 128 {
            todo!("DeltaNet norm gate kernel supports head_v_dim <= 128, got {}", config.value_head_dim);
        }

        let mixer_tree = resolve_subtree(decoder_layer_loader, &["mixer"]);
        let conv_tree = resolve_subtree(&mixer_tree, &["conv", "conv1d"]);

        let data_type: DataType = config.in_proj_config.activation_precision().into();

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
        .expect("Failed to create in-projection kernel");

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
        .expect("Failed to create out-projection kernel");

        let conv_weight = conv_tree.leaf_array("weights").unwrap();
        let conv_bias = if config.conv_config.has_biases {
            Some(conv_tree.leaf_array("biases").unwrap())
        } else {
            None
        };

        let a_log = mixer_tree.leaf_array("a_log").unwrap();
        let dt_bias = mixer_tree.leaf_array("dt_bias").unwrap();
        let norm_weight = resolve_subtree(&mixer_tree, &["norm", "inner_norm"]).leaf_array("scales").unwrap();

        let has_bias = config.conv_config.has_biases;

        let conv_update = <B::Kernels as Kernels>::DeltaNetConvUpdateKernel::new(context, data_type, has_bias)
            .expect("Failed to create DeltaNet conv update kernel");
        let delta_net_update =
            <B::Kernels as Kernels>::DeltaNetUpdateKernel::new(context, data_type, config.head_dim as u32)
                .expect("Failed to create DeltaNet update kernel");

        let conv_pack = <B::Kernels as Kernels>::Conv1dPackKernel::new(context, data_type)
            .expect("Failed to create Conv1dPack kernel");
        let conv_scan = <B::Kernels as Kernels>::DeltaNetConvScanKernel::new(context, data_type, has_bias)
            .expect("Failed to create DeltaNet conv scan kernel");
        let prefill_prep =
            <B::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(context, data_type, config.head_dim as u32)
                .expect("Failed to create DeltaNet prefill prep kernel");
        let delta_net_prefill =
            <B::Kernels as Kernels>::DeltaNetPrefillKernel::new(context, data_type, config.head_dim as u32)
                .expect("Failed to create DeltaNet prefill kernel");
        let norm_gate = <B::Kernels as Kernels>::DeltaNetNormGateKernel::new(context, data_type)
            .expect("Failed to create DeltaNet norm gate kernel");

        Self {
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
        }
    }

    fn run_conv_update(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) {
        let in_proj = state.array(ArrayId::SsmInProj);
        let conv_state = state.array(ArrayId::DeltaNetConvState(self.layer_index));

        let weight_buf_rc = self.conv_weight.buffer();
        let weight_buf_borrow = weight_buf_rc.borrow();
        let bias_buf_rc = self.conv_bias.as_ref().map(|b| b.buffer());
        let bias_buf_borrow = bias_buf_rc.as_ref().map(|rc| rc.borrow());

        let kernel_size = self.config.kernel_size;

        self.conv_update.encode(
            weight_buf_borrow.deref(),
            bias_buf_borrow.as_deref(),
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

        let weight_buf_rc = self.conv_weight.buffer();
        let weight_buf_borrow = weight_buf_rc.borrow();
        let bias_buf_rc = self.conv_bias.as_ref().map(|b| b.buffer());
        let bias_buf_borrow = bias_buf_rc.as_ref().map(|rc| rc.borrow());

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
            weight_buf_borrow.deref(),
            bias_buf_borrow.as_deref(),
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

        let a_log_rc = self.a_log.buffer();
        let a_log_borrow = a_log_rc.borrow();
        let dt_bias_rc = self.dt_bias.buffer();
        let dt_bias_borrow = dt_bias_rc.borrow();
        let norm_weight_rc = self.norm_weight.buffer();
        let norm_weight_borrow = norm_weight_rc.borrow();

        self.delta_net_update.encode(
            in_proj.buffer().borrow().deref(),
            a_log_borrow.deref(),
            dt_bias_borrow.deref(),
            norm_weight_borrow.deref(),
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

        let a_log_rc = self.a_log.buffer();
        let a_log_borrow = a_log_rc.borrow();
        let dt_bias_rc = self.dt_bias.buffer();
        let dt_bias_borrow = dt_bias_rc.borrow();
        let norm_weight_rc = self.norm_weight.buffer();
        let norm_weight_borrow = norm_weight_rc.borrow();

        // Prep buffers: pre-allocated in ScratchBuffers, reused across layers
        let (prep_q_norm, prep_k_norm, prep_beta, prep_decay) =
            state.delta_net_prep_buffers().expect("DeltaNet prep buffers not initialized");

        self.prefill_prep.encode(
            in_proj.buffer().borrow().deref(),
            a_log_borrow.deref(),
            dt_bias_borrow.deref(),
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
            norm_weight_borrow.deref(),
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
        let active_suffix_length = state.active_suffix_length();
        if active_suffix_length == 0 {
            return Ok(());
        }

        self.in_projection.encode(state, encoder)?;

        if active_suffix_length == 1 {
            self.run_conv_update(state, encoder);
            self.run_delta_rule(state, encoder);
        } else {
            self.run_conv_scan(state, encoder, active_suffix_length);
            self.run_delta_rule_prefill(state, encoder, active_suffix_length);
        }

        self.out_projection.encode(state, encoder)?;
        Ok(())
    }
}
