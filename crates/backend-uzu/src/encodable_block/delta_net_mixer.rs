use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        kernel::{
            Conv1dPackKernel, DeltaNetConvScanKernel, DeltaNetConvUpdateKernel, DeltaNetNormGateKernel,
            DeltaNetPrefillKernel, DeltaNetPrefillPrepKernel, DeltaNetUpdateKernel,
        },
    },
    config::token_mixer::delta_net::DeltaNetConfig,
    data_type::DataType,
    encodable_block::linear::{Linear, LinearBlockError},
    forward_pass::delta_net_layer::DeltaNetLayer,
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum DeltaNetMixerError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Unsupported configuration: {0}")]
    UnsupportedConfiguration(String),
    #[error("Linear error: {0}")]
    InnerLinearError(#[from] LinearBlockError<B>),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
}

pub struct DeltaNetMixer<B: Backend> {
    kernel_size: usize,
    num_heads: usize,
    num_groups: usize,
    value_head_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    total_proj_dim: usize,
    norm_epsilon: f32,
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
    inner_data_type: DataType,
}

pub struct DeltaNetArguments<'a, B: Backend> {
    pub active_row_count: usize,
    pub layer: &'a mut DeltaNetLayer<B>,
}

impl<B: Backend> DeltaNetMixer<B> {
    pub fn new(
        context: &B::Context,
        config: &DeltaNetConfig,
        model_dim: usize,
        parameter_tree: &ParameterTree<B>,
        outer_data_type: DataType,
    ) -> Result<(Self, Option<Allocation<B>>), DeltaNetMixerError<B>> {
        let inner_data_type = DataType::F32;

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

        let has_bias = config.conv_config.has_biases;
        let key_dim = config.key_dim();
        let value_dim = config.value_dim();
        let conv_dim = config.conv_dim();
        let total_proj_dim = config.total_proj_dim();

        // Load weights
        let conv_tree = parameter_tree.subtree("conv")?;

        let (in_projection, in_projection_input_hadamard_factors) =
            <dyn Linear<B>>::new_extracting_input_hadamard_mixed_precision(
                model_dim,
                [total_proj_dim],
                false,
                context,
                outer_data_type,
                outer_data_type,
                inner_data_type,
                &parameter_tree.subtree("in_proj")?,
            )?;

        let out_projection = <dyn Linear<B>>::new_mixed_precision(
            value_dim,
            [model_dim],
            false,
            context,
            outer_data_type,
            inner_data_type,
            outer_data_type,
            &parameter_tree.subtree("out_proj")?,
        )?;

        let conv_weight =
            conv_tree.leaf("weights")?.validate(&[conv_dim, config.kernel_size], inner_data_type)?.read_allocation()?;
        let conv_bias = if has_bias {
            Some(conv_tree.leaf("biases")?.validate(&[conv_dim], inner_data_type)?.read_allocation()?)
        } else {
            None
        };

        let a_log = parameter_tree.leaf("a_log")?.validate(&[config.num_heads], inner_data_type)?.read_allocation()?;
        let dt_bias =
            parameter_tree.leaf("dt_bias")?.validate(&[config.num_heads], inner_data_type)?.read_allocation()?;
        let norm_weight = parameter_tree
            .leaf("norm.scales")?
            .validate(&[config.value_head_dim], inner_data_type)?
            .read_allocation()?;

        // Create kernels
        let conv_update = <B::Kernels as Kernels>::DeltaNetConvUpdateKernel::new(context, inner_data_type, has_bias)
            .map_err(DeltaNetMixerError::BackendError)?;
        let delta_net_update =
            <B::Kernels as Kernels>::DeltaNetUpdateKernel::new(context, inner_data_type, config.head_dim as u32)
                .map_err(DeltaNetMixerError::BackendError)?;
        let conv_pack = <B::Kernels as Kernels>::Conv1dPackKernel::new(context, inner_data_type)
            .map_err(DeltaNetMixerError::BackendError)?;
        let conv_scan = <B::Kernels as Kernels>::DeltaNetConvScanKernel::new(context, inner_data_type, has_bias)
            .map_err(DeltaNetMixerError::BackendError)?;
        let prefill_prep =
            <B::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(context, inner_data_type, config.head_dim as u32)
                .map_err(DeltaNetMixerError::BackendError)?;
        let delta_net_prefill =
            <B::Kernels as Kernels>::DeltaNetPrefillKernel::new(context, inner_data_type, config.head_dim as u32)
                .map_err(DeltaNetMixerError::BackendError)?;
        let norm_gate = <B::Kernels as Kernels>::DeltaNetNormGateKernel::new(context, inner_data_type)
            .map_err(DeltaNetMixerError::BackendError)?;

        Ok((
            Self {
                kernel_size: config.kernel_size,
                num_heads: config.num_heads,
                num_groups: config.num_groups,
                value_head_dim: config.value_head_dim,
                key_dim,
                value_dim,
                conv_dim,
                total_proj_dim,
                norm_epsilon: config.norm_config.epsilon,
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
                inner_data_type,
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
        let kernel_size = self.kernel_size;
        self.conv_update.encode(
            &self.conv_weight,
            self.conv_bias.as_ref(),
            in_proj,
            &mut layer.conv_state,
            kernel_size as u32,
            self.conv_dim as u32,
            (kernel_size - 1) as u32,
            encoder,
        );
    }

    fn run_conv_scan(
        &self,
        layer: &mut DeltaNetLayer<B>,
        in_proj: &mut Allocation<B>,
        encoder: &mut Encoder<B>,
        suffix_length: usize,
    ) -> Result<(), B::Error> {
        let kernel_size = self.kernel_size;
        let state_stride = kernel_size - 1;
        let conv_dim = self.conv_dim;
        let total_proj_dim = self.total_proj_dim;
        let mut padded = encoder
            .allocate_scratch(size_for_shape(&[suffix_length + state_stride, total_proj_dim], self.inner_data_type))?;

        self.conv_pack.encode(
            &layer.conv_state,
            &*in_proj,
            &mut padded,
            state_stride as u32,
            total_proj_dim as u32,
            suffix_length as u32,
            conv_dim as u32,
            encoder,
        );

        self.conv_scan.encode(
            &padded,
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
        Ok(())
    }

    fn run_delta_rule(
        &self,
        layer: &mut DeltaNetLayer<B>,
        in_proj: &Allocation<B>,
        active_row_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut out =
            encoder.allocate_scratch(size_for_shape(&[active_row_count, self.value_dim], self.inner_data_type))?;
        self.delta_net_update.encode(
            in_proj,
            &self.a_log,
            &self.dt_bias,
            &self.norm_weight,
            &mut layer.ssm_state,
            &mut out,
            self.num_heads as u32,
            self.num_groups as u32,
            self.value_head_dim as u32,
            self.key_dim as u32,
            self.value_dim as u32,
            self.norm_epsilon,
            encoder,
        );
        Ok(out)
    }

    fn run_delta_rule_prefill(
        &self,
        layer: &mut DeltaNetLayer<B>,
        in_proj: &Allocation<B>,
        suffix_length: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let num_dv_groups = ((self.value_head_dim + 7) / 8) as u32;
        let mut prep_q_norm =
            encoder.allocate_scratch(size_for_shape(&[suffix_length * self.key_dim], DataType::F32))?;
        let mut prep_k_norm =
            encoder.allocate_scratch(size_for_shape(&[suffix_length * self.key_dim], DataType::F32))?;
        let mut prep_beta =
            encoder.allocate_scratch(size_for_shape(&[suffix_length * self.num_heads], DataType::F32))?;
        let mut prep_decay =
            encoder.allocate_scratch(size_for_shape(&[suffix_length * self.num_heads], DataType::F32))?;

        self.prefill_prep.encode(
            in_proj,
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
            suffix_length as u32,
            encoder,
        );

        let mut out =
            encoder.allocate_scratch(size_for_shape(&[suffix_length, self.value_dim], self.inner_data_type))?;
        self.delta_net_prefill.encode(
            &prep_q_norm,
            &prep_k_norm,
            &prep_beta,
            &prep_decay,
            in_proj,
            &mut layer.ssm_state,
            &mut out,
            self.num_heads as u32,
            self.num_groups as u32,
            self.value_head_dim as u32,
            self.key_dim as u32,
            self.value_dim as u32,
            suffix_length as u32,
            num_dv_groups,
            encoder,
        );

        self.norm_gate.encode(
            &mut out,
            in_proj,
            &self.norm_weight,
            self.num_heads as u32,
            self.value_head_dim as u32,
            self.value_dim as u32,
            self.conv_dim as u32,
            self.total_proj_dim as u32,
            self.norm_epsilon,
            suffix_length as u32,
            encoder,
        );
        Ok(out)
    }

    pub fn encode(
        &self,
        args: DeltaNetArguments<B>,
        input: Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let DeltaNetArguments {
            active_row_count,
            layer,
        } = args;
        assert!(active_row_count > 0, "DeltaNet mixer requires at least one active row");

        let mut in_proj = self.in_projection.encode(input, active_row_count, encoder)?;

        let delta_output = if active_row_count == 1 {
            self.run_conv_update(layer, &mut in_proj, encoder);
            self.run_delta_rule(layer, &in_proj, active_row_count, encoder)?
        } else {
            self.run_conv_scan(layer, &mut in_proj, encoder, active_row_count)?;
            self.run_delta_rule_prefill(layer, &in_proj, active_row_count, encoder)?
        };

        self.out_projection.encode(delta_output, active_row_count, encoder)
    }
}
