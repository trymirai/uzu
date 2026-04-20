use thiserror::Error;

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        kernel::{
            Conv1dPackKernel, DeltaNetConvScanKernel, DeltaNetConvUpdateKernel, DeltaNetNormGateKernel,
            DeltaNetPrefillKernel, DeltaNetPrefillPrepKernel, DeltaNetUpdateKernel,
        },
    },
    config::DeltaNetAttentionConfig,
    encodable_block::linear::{Linear, LinearBlockError},
    forward_pass::delta_net_layer::DeltaNetLayer,
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
    conv_weight: Allocation<B>,
    conv_bias: Option<Allocation<B>>,
    a_log: Allocation<B>,
    dt_bias: Allocation<B>,
    norm_weight: Allocation<B>,
    data_type: DataType,
}

pub(crate) struct DeltaNetArguments<'a, B: Backend> {
    pub context: &'a B::Context,
    pub active_row_count: usize,
    pub layer: &'a mut DeltaNetLayer<B>,
}

impl<B: Backend> DeltaNetMixer<B> {
    pub(crate) fn new(
        context: &B::Context,
        config: DeltaNetAttentionConfig,
        model_dim: usize,
        decoder_layer_loader: &ParameterTree<B::Context>,
    ) -> Result<(Self, Option<Allocation<B>>), DeltaNetMixerError<B>> {
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

        let (in_projection, in_projection_input_hadamard_factors) = <dyn Linear<B>>::new_extracting_input_hadamard(
            &config.in_proj_config,
            model_dim,
            [config.total_proj_dim()],
            context,
            &resolve_subtree(decoder_layer_loader, &["mixer.in_projection", "mixer.in_proj"]),
        )
        .map_err(|e| DeltaNetMixerError::InnerLinearError(Box::new(e)))?;

        let out_projection = <dyn Linear<B>>::new(
            &config.out_proj_config,
            config.value_dim(),
            [model_dim],
            context,
            &resolve_subtree(decoder_layer_loader, &["mixer.out_projection", "mixer.out_proj"]),
        )
        .map_err(|e| DeltaNetMixerError::InnerLinearError(Box::new(e)))?;

        let conv_weight = conv_tree.leaf("weights")?.read_allocation()?;
        let conv_bias = if has_bias {
            Some(conv_tree.leaf("biases")?.read_allocation()?)
        } else {
            None
        };

        let a_log = mixer_tree.leaf("a_log")?.read_allocation()?;
        let dt_bias = mixer_tree.leaf("dt_bias")?.read_allocation()?;
        let norm_tree = resolve_subtree(&mixer_tree, &["norm", "inner_norm"]);
        let norm_weight = norm_tree.leaf("scales")?.read_allocation()?;

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

        Ok((
            Self {
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
                data_type,
            },
            in_projection_input_hadamard_factors,
        ))
    }

    fn run_conv_update(
        &self,
        layer: &mut DeltaNetLayer<B>,
        in_proj: &mut Allocation<B>,
        encoder: &mut Encoder<B>,
    ) {
        let kernel_size = self.config.kernel_size;
        self.conv_update.encode(
            &self.conv_weight,
            self.conv_bias.as_ref(),
            in_proj,
            &mut layer.conv_state,
            kernel_size as u32,
            self.config.conv_dim() as u32,
            (kernel_size - 1) as u32,
            encoder,
        );
    }

    fn run_conv_scan(
        &self,
        layer: &mut DeltaNetLayer<B>,
        in_proj: &mut Allocation<B>,
        padded: &mut Allocation<B>,
        encoder: &mut Encoder<B>,
        suffix_length: usize,
    ) {
        let kernel_size = self.config.kernel_size;
        let state_stride = kernel_size - 1;
        let conv_dim = self.config.conv_dim();
        let total_proj_dim = self.config.total_proj_dim();

        self.conv_pack.encode(
            &layer.conv_state,
            &*in_proj,
            &mut *padded,
            state_stride as u32,
            total_proj_dim as u32,
            suffix_length as u32,
            conv_dim as u32,
            encoder,
        );

        self.conv_scan.encode(
            &*padded,
            &self.conv_weight,
            self.conv_bias.as_ref(),
            in_proj,
            &mut layer.conv_state,
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
        layer: &mut DeltaNetLayer<B>,
        in_proj: &Allocation<B>,
        out: &mut Allocation<B>,
        encoder: &mut Encoder<B>,
    ) {
        self.delta_net_update.encode(
            in_proj,
            &self.a_log,
            &self.dt_bias,
            &self.norm_weight,
            &mut layer.ssm_state,
            out,
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
        layer: &mut DeltaNetLayer<B>,
        in_proj: &Allocation<B>,
        out: &mut Allocation<B>,
        prep_q_norm: &mut Allocation<B>,
        prep_k_norm: &mut Allocation<B>,
        prep_beta: &mut Allocation<B>,
        prep_decay: &mut Allocation<B>,
        encoder: &mut Encoder<B>,
        suffix_length: usize,
    ) {
        let num_dv_groups = ((self.config.value_head_dim + 7) / 8) as u32;

        self.prefill_prep.encode(
            in_proj,
            &self.a_log,
            &self.dt_bias,
            &mut *prep_q_norm,
            &mut *prep_k_norm,
            &mut *prep_beta,
            &mut *prep_decay,
            self.config.num_heads as u32,
            self.config.num_groups as u32,
            self.config.key_dim() as u32,
            self.config.value_dim() as u32,
            suffix_length as u32,
            encoder,
        );

        self.delta_net_prefill.encode(
            &*prep_q_norm,
            &*prep_k_norm,
            &*prep_beta,
            &*prep_decay,
            in_proj,
            &mut layer.ssm_state,
            &mut *out,
            self.config.num_heads as u32,
            self.config.num_groups as u32,
            self.config.value_head_dim as u32,
            self.config.key_dim() as u32,
            self.config.value_dim() as u32,
            suffix_length as u32,
            num_dv_groups,
            encoder,
        );

        self.norm_gate.encode(
            out,
            in_proj,
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
        args: DeltaNetArguments<'_, B>,
        input: &mut Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let DeltaNetArguments {
            context,
            active_row_count,
            layer,
        } = args;
        assert!(active_row_count > 0, "DeltaNet mixer requires at least one active row");

        let mut in_proj = self.in_projection.encode(context, input, active_row_count, encoder)?;
        let mut delta_output =
            encoder.allocate_scratch(size_for_shape(&[active_row_count, self.config.value_dim()], self.data_type))?;

        if active_row_count == 1 {
            self.run_conv_update(layer, &mut in_proj, encoder);
            self.run_delta_rule(layer, &in_proj, &mut delta_output, encoder);
        } else {
            let state_stride = self.config.kernel_size.saturating_sub(1);
            let mut padded = encoder.allocate_scratch(size_for_shape(
                &[active_row_count + state_stride, self.config.total_proj_dim()],
                self.data_type,
            ))?;
            self.run_conv_scan(layer, &mut in_proj, &mut padded, encoder, active_row_count);

            let mut prep_q_norm =
                encoder.allocate_scratch(size_for_shape(&[active_row_count * self.config.key_dim()], DataType::F32))?;
            let mut prep_k_norm =
                encoder.allocate_scratch(size_for_shape(&[active_row_count * self.config.key_dim()], DataType::F32))?;
            let mut prep_beta =
                encoder.allocate_scratch(size_for_shape(&[active_row_count * self.config.num_heads], DataType::F32))?;
            let mut prep_decay =
                encoder.allocate_scratch(size_for_shape(&[active_row_count * self.config.num_heads], DataType::F32))?;
            self.run_delta_rule_prefill(
                layer,
                &in_proj,
                &mut delta_output,
                &mut prep_q_norm,
                &mut prep_k_norm,
                &mut prep_beta,
                &mut prep_decay,
                encoder,
                active_row_count,
            );
        }

        self.out_projection.encode(context, &mut delta_output, active_row_count, encoder)
    }
}
