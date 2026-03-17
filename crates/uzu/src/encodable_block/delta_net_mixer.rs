use std::ops::{Deref, DerefMut};

use crate::{
    DataType,
    array::Array,
    backends::common::{
        Backend, CommandBuffer, Kernels,
        kernel::{DeltaNetConvUpdateKernel, DeltaNetUpdateKernel},
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
    conv_update: <B::Kernels as Kernels>::DeltaNetConvUpdateKernel,
    delta_net_update: <B::Kernels as Kernels>::DeltaNetUpdateKernel,
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

        let mixer_tree = resolve_subtree(decoder_layer_loader, &["mixer"]);
        let conv_tree = resolve_subtree(&mixer_tree, &["conv", "conv1d"]);

        let data_type: DataType = config.in_proj_config.activation_precision().into();

        let value_dim = config.value_dim();
        let conv_dim = config.conv_dim();
        // Total in_proj output: conv_dim + value_dim (z) + num_heads (beta) + num_heads (a)
        let total_proj_dim = conv_dim + value_dim + config.num_heads + config.num_heads;

        let in_projection = <dyn Linear<B>>::new(
            &config.in_proj_config,
            false,
            model_dim,
            [total_proj_dim],
            context,
            &resolve_subtree(decoder_layer_loader, &["mixer.in_projection", "mixer.in_proj"]),
            ArrayId::Main,
            ArrayId::SsmInProj,
        )
        .expect("Failed to create in-projection kernel");

        let out_projection = <dyn Linear<B>>::new(
            &config.out_proj_config,
            false,
            value_dim,
            [model_dim],
            context,
            &resolve_subtree(decoder_layer_loader, &["mixer.out_projection", "mixer.out_proj"]),
            ArrayId::AttentionOutput,
            ArrayId::Main,
        )
        .expect("Failed to create out-projection kernel");

        let conv_weight = conv_tree.leaf_array("weights").unwrap().clone();
        let conv_bias = if config.conv_config.has_biases {
            Some(conv_tree.leaf_array("biases").unwrap().clone())
        } else {
            None
        };

        let a_log = mixer_tree.leaf_array("a_log").unwrap().clone();
        let dt_bias = mixer_tree.leaf_array("dt_bias").unwrap().clone();
        let norm_weight = resolve_subtree(&mixer_tree, &["norm", "inner_norm"]).leaf_array("weights").unwrap().clone();

        let has_bias = config.conv_config.has_biases;
        let conv_update = <B::Kernels as Kernels>::DeltaNetConvUpdateKernel::new(context, data_type, has_bias)
            .expect("Failed to create DeltaNet conv update kernel");
        let delta_net_update = <B::Kernels as Kernels>::DeltaNetUpdateKernel::new(context, data_type)
            .expect("Failed to create fused DeltaNet decode kernel");

        Self {
            layer_index,
            config,
            in_projection,
            out_projection,
            conv_update,
            delta_net_update,
            conv_weight,
            conv_bias,
            a_log,
            dt_bias,
            norm_weight,
        }
    }

    // TODO: add prefill path (conv_scan + chunked delta rule)
    fn run_conv(
        &self,
        state: &mut ForwardPassState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        let arrays = state.arrays(&[ArrayId::SsmInProj, ArrayId::DeltaNetConvState(self.layer_index)]);
        let in_proj = arrays[0].borrow();
        let conv_state = arrays[1].borrow();

        let weight_buf_rc = self.conv_weight.buffer();
        let weight_buf_borrow = weight_buf_rc.borrow();
        let bias_buf_rc = self.conv_bias.as_ref().map(|b| b.buffer());
        let bias_buf_borrow = bias_buf_rc.as_ref().map(|rc| rc.borrow());

        let conv_dim = self.config.conv_dim();
        let kernel_size = self.config.kernel_size;
        let state_stride = kernel_size.saturating_sub(1);

        self.conv_update.encode(
            weight_buf_borrow.deref(),
            bias_buf_borrow.as_deref(),
            in_proj.buffer().borrow_mut().deref_mut(),
            conv_state.buffer().borrow_mut().deref_mut(),
            kernel_size as u32,
            conv_dim as u32,
            state_stride as u32,
            command_buffer,
        );
    }

    fn run_delta_rule(
        &self,
        state: &mut ForwardPassState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        let arrays =
            state.arrays(&[ArrayId::SsmInProj, ArrayId::DeltaNetSsmState(self.layer_index), ArrayId::AttentionOutput]);
        let in_proj = arrays[0].borrow();
        let ssm_state = arrays[1].borrow();
        let out = arrays[2].borrow();

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
            self.config.head_dim as u32,
            self.config.value_head_dim as u32,
            self.config.key_dim() as u32,
            self.config.value_dim() as u32,
            self.config.norm_config.epsilon,
            command_buffer,
        );
    }

    pub(crate) fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let active_suffix_length = state.active_suffix_length();
        if active_suffix_length == 0 {
            return Ok(());
        }

        self.in_projection.encode(state, command_buffer)?;
        self.run_conv(state, command_buffer);
        self.run_delta_rule(state, command_buffer);
        self.out_projection.encode(state, command_buffer)?;
        Ok(())
    }
}
