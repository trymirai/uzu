//! Mamba2 SSM mixer encodable.

mod ssd_prefill;

use std::env;

use ssd_prefill::{SSDPrefillArguments, SSDPrefillBlock, SSDPrefillMode};
use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::ActivationType,
        kernel::{Conv1dDecodeKernel, Conv1dPackKernel, Conv1dScanKernel, SSDUpdateKernel, SplitInProjKernel},
    },
    config::token_mixer::mamba2::Mamba2Config,
    data_type::DataType,
    encodable_block::linear::{Linear, LinearBlockError},
    forward_pass::ssm_layer::SSMLayer,
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum MambaMixerError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Linear error: {0}")]
    LinearError(#[from] LinearBlockError<B>),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
}

pub struct MambaMixer<B: Backend> {
    conv_dim: usize,
    inner_dim: usize,
    num_heads: usize,
    num_groups: usize,
    head_dim: usize,
    state_dim: usize,
    kernel_size: usize,
    activation_type: ActivationType,
    in_projection: Box<dyn Linear<B>>,
    out_projection: Box<dyn Linear<B>>,
    split_inproj: <B::Kernels as Kernels>::SplitInProjKernel,
    conv_decode: <B::Kernels as Kernels>::Conv1dDecodeKernel,
    conv_pack: <B::Kernels as Kernels>::Conv1dPackKernel,
    conv_scan: <B::Kernels as Kernels>::Conv1dScanKernel,
    ssd_prefill: SSDPrefillBlock<B>,
    ssd_update: <B::Kernels as Kernels>::SSDUpdateKernel,
    conv_weight: Allocation<B>,
    conv_bias: Option<Allocation<B>>,
    gate_bias: Allocation<B>,
    skip_connection_weight: Allocation<B>,
    prefill_mode: SSDPrefillMode,
    inner_data_type: DataType,
}

pub struct MambaArguments<'a, B: Backend> {
    pub active_row_count: usize,
    pub layer: &'a mut SSMLayer<B>,
}

struct SplitInProjectionOutput<B: Backend> {
    conv_inputs: Allocation<B>,
    gate: Allocation<B>,
    time_step: Allocation<B>,
}

struct ConvScanOutput<B: Backend> {
    conv_x: Allocation<B>,
    state_b: Allocation<B>,
    state_c: Allocation<B>,
}

fn resolve_prefill_mode_from_env() -> SSDPrefillMode {
    match env::var("UZU_SSM_PREFILL_MODE") {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "seq" | "sequential" | "baseline" => SSDPrefillMode::Sequential,
            "single" | "singlepass" | "single_pass" => SSDPrefillMode::SinglePass,
            _ => SSDPrefillMode::SinglePass,
        },
        Err(_) => SSDPrefillMode::SinglePass,
    }
}

impl<B: Backend> MambaMixer<B> {
    pub fn new(
        context: &B::Context,
        mamba_config: &Mamba2Config,
        model_dim: usize,
        parameter_tree: &ParameterTree<B>,
        outer_data_type: DataType,
    ) -> Result<(Self, Option<Allocation<B>>), MambaMixerError<B>> {
        let inner_data_type = DataType::F32;

        let conv_tree = parameter_tree.subtree("conv")?;
        let conv_dim = mamba_config.conv_dim();
        let inner_dim = mamba_config.inner_dim();

        let (in_projection, in_projection_input_hadamard_factors) =
            <dyn Linear<B>>::new_extracting_input_hadamard_mixed_precision(
                model_dim,
                [conv_dim, inner_dim, mamba_config.num_heads],
                mamba_config.has_in_biases,
                context,
                outer_data_type,
                outer_data_type,
                inner_data_type,
                &parameter_tree.subtree("in_projection")?,
            )?;

        let out_projection = <dyn Linear<B>>::new_mixed_precision(
            inner_dim,
            [model_dim],
            mamba_config.has_out_biases,
            context,
            outer_data_type,
            inner_data_type,
            outer_data_type,
            &parameter_tree.subtree("out_projection")?,
        )?;

        let conv_weight = conv_tree
            .leaf("weights")?
            .validate(&[conv_dim, mamba_config.kernel_size], inner_data_type)?
            .read_allocation()?;
        let conv_bias = if mamba_config.conv_config.has_biases {
            Some(conv_tree.leaf("biases")?.validate(&[conv_dim], inner_data_type)?.read_allocation()?)
        } else {
            None
        };
        let gate_bias = parameter_tree.leaf("gate_bias")?.validate(&[inner_dim], inner_data_type)?.read_allocation()?;
        let skip_connection_weight = parameter_tree
            .leaf("skip_connection_weight")?
            .validate(&[mamba_config.num_heads], inner_data_type)?
            .read_allocation()?;

        let split_inproj = <B::Kernels as Kernels>::SplitInProjKernel::new(context, inner_data_type)
            .map_err(MambaMixerError::BackendError)?;
        let conv_scan = <B::Kernels as Kernels>::Conv1dScanKernel::new(
            context,
            inner_data_type,
            mamba_config.conv_config.has_biases,
        )
        .map_err(MambaMixerError::BackendError)?;
        let conv_decode = <B::Kernels as Kernels>::Conv1dDecodeKernel::new(
            context,
            inner_data_type,
            mamba_config.conv_config.has_biases,
            true,
        )
        .map_err(MambaMixerError::BackendError)?;
        let conv_pack = <B::Kernels as Kernels>::Conv1dPackKernel::new(context, inner_data_type)
            .map_err(MambaMixerError::BackendError)?;
        let ssd_prefill = SSDPrefillBlock::new(context, inner_data_type).map_err(MambaMixerError::BackendError)?;
        let ssd_update = <B::Kernels as Kernels>::SSDUpdateKernel::new(context, inner_data_type, true)
            .map_err(MambaMixerError::BackendError)?;
        let prefill_mode = resolve_prefill_mode_from_env();

        Ok((
            Self {
                conv_dim,
                inner_dim,
                num_heads: mamba_config.num_heads,
                num_groups: mamba_config.num_groups,
                head_dim: mamba_config.head_dim,
                state_dim: mamba_config.state_dim,
                kernel_size: mamba_config.kernel_size,
                activation_type: mamba_config.activation.act_type(),
                in_projection,
                out_projection,
                split_inproj,
                conv_decode,
                conv_pack,
                conv_scan,
                ssd_prefill,
                ssd_update,
                conv_weight,
                conv_bias,
                gate_bias,
                skip_connection_weight,
                prefill_mode,
                inner_data_type,
            },
            in_projection_input_hadamard_factors,
        ))
    }

    fn run_split_inproj(
        &self,
        in_proj: &Allocation<B>,
        suffix_length: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<SplitInProjectionOutput<B>, B::Error> {
        let conv_dim = self.conv_dim;
        let inner_dim = self.inner_dim;
        let num_heads = self.num_heads;
        let total_dim = conv_dim + inner_dim + num_heads;
        let mut conv_inputs =
            encoder.allocate_scratch(size_for_shape(&[suffix_length, conv_dim], self.inner_data_type))?;
        let mut gate = encoder
            .allocate_scratch(size_for_shape(&[suffix_length, num_heads, self.head_dim], self.inner_data_type))?;
        let mut time_step =
            encoder.allocate_scratch(size_for_shape(&[suffix_length, num_heads], self.inner_data_type))?;
        self.split_inproj.encode(
            in_proj,
            &mut conv_inputs,
            &mut gate,
            &mut time_step,
            &self.gate_bias,
            suffix_length as u32,
            total_dim as u32,
            conv_dim as u32,
            inner_dim as u32,
            num_heads as u32,
            encoder,
        );
        Ok(SplitInProjectionOutput {
            conv_inputs,
            gate,
            time_step,
        })
    }

    fn run_conv_scan(
        &self,
        layer: &mut SSMLayer<B>,
        conv_inputs: &Allocation<B>,
        encoder: &mut Encoder<B>,
        suffix_length: usize,
    ) -> Result<ConvScanOutput<B>, B::Error> {
        let conv_dim = self.conv_dim;
        let inner_dim = self.inner_dim;
        let proj_dim = self.num_groups * self.state_dim;
        let state_stride = self.kernel_size.saturating_sub(1);
        let mut conv_x = encoder
            .allocate_scratch(size_for_shape(&[suffix_length, self.num_heads, self.head_dim], self.inner_data_type))?;
        let state_shape = [suffix_length, self.num_groups, self.state_dim];
        let mut state_b = encoder.allocate_scratch(size_for_shape(&state_shape, self.inner_data_type))?;
        let mut state_c = encoder.allocate_scratch(size_for_shape(&state_shape, self.inner_data_type))?;

        if suffix_length == 1 {
            if conv_dim > 0 && self.kernel_size > 0 {
                let next_state = layer.conv_state.as_mut().unwrap_or(&mut layer.ssm_state);
                self.conv_decode.encode(
                    conv_inputs,
                    &self.conv_weight,
                    self.conv_bias.as_ref(),
                    None::<&Allocation<B>>,
                    &mut conv_x,
                    &mut state_b,
                    &mut state_c,
                    next_state,
                    self.kernel_size as u32,
                    conv_dim as u32,
                    state_stride as u32,
                    conv_dim as u32,
                    suffix_length as u32,
                    inner_dim as u32,
                    proj_dim as u32,
                    self.activation_type,
                    encoder,
                );
            }
        } else {
            let mut padded = (conv_dim > 0 && state_stride > 0)
                .then(|| {
                    encoder.allocate_scratch(size_for_shape(
                        &[suffix_length + state_stride, conv_dim],
                        self.inner_data_type,
                    ))
                })
                .transpose()?;
            if let Some(padded) = padded.as_mut() {
                let state_in = layer.conv_state.as_ref().unwrap_or(conv_inputs);
                self.conv_pack.encode(
                    state_in,
                    conv_inputs,
                    padded,
                    state_stride as u32,
                    conv_dim as u32,
                    suffix_length as u32,
                    conv_dim as u32,
                    encoder,
                );
            }

            if conv_dim > 0 && self.kernel_size > 0 {
                let conv_source = match &padded {
                    Some(padded) => padded,
                    None => conv_inputs,
                };
                let next_state = layer.conv_state.as_mut().unwrap_or(&mut layer.ssm_state);
                self.conv_scan.encode(
                    conv_source,
                    &self.conv_weight,
                    self.conv_bias.as_ref(),
                    &mut conv_x,
                    &mut state_b,
                    &mut state_c,
                    next_state,
                    suffix_length as u32,
                    self.kernel_size as u32,
                    conv_dim as u32,
                    state_stride as u32,
                    conv_dim as u32,
                    inner_dim as u32,
                    proj_dim as u32,
                    self.activation_type,
                    encoder,
                );
            }
        }
        Ok(ConvScanOutput {
            conv_x,
            state_b,
            state_c,
        })
    }

    fn run_prefill_ssm(
        &self,
        layer: &mut SSMLayer<B>,
        conv_x: &Allocation<B>,
        state_b: &Allocation<B>,
        state_c: &Allocation<B>,
        time_step: &Allocation<B>,
        gate: &Allocation<B>,
        encoder: &mut Encoder<B>,
        suffix_length: usize,
    ) -> Result<Allocation<B>, B::Error> {
        self.ssd_prefill.encode(
            encoder,
            SSDPrefillArguments {
                x: conv_x,
                dt: time_step,
                b: state_b,
                c: state_c,
                d: &self.skip_connection_weight,
                z: gate,
                state: &mut layer.ssm_state,
                suffix_len: suffix_length,
                group_size: (self.num_heads / self.num_groups) as u32,
                state_size: self.state_dim as u32,
                x_strides: [self.num_heads * self.head_dim, self.head_dim, 1],
                dt_strides: [self.num_heads, 1],
                cb_strides: [self.num_groups * self.state_dim, self.state_dim, 1],
                state_strides: [self.head_dim * self.state_dim, self.state_dim, 1],
                channels: self.num_heads,
                head_dim: self.head_dim,
            },
            self.prefill_mode,
        )
    }

    fn run_decode_ssm(
        &self,
        layer: &mut SSMLayer<B>,
        conv_x: &Allocation<B>,
        state_b: &Allocation<B>,
        state_c: &Allocation<B>,
        time_step: &Allocation<B>,
        gate: &Allocation<B>,
        encoder: &mut Encoder<B>,
        suffix_length: usize,
    ) -> Result<Allocation<B>, B::Error> {
        let mut out =
            encoder.allocate_scratch(size_for_shape(&[suffix_length, self.inner_dim], self.inner_data_type))?;
        let h = self.num_heads as u32;
        let g = self.num_groups as u32;
        let dh = self.head_dim as u32;
        let n = self.state_dim as u32;

        let x_strides = [h * dh, dh, 1u32];
        let dt_strides = [h, 1u32];
        let cb_strides = [g * n, n, 1u32];
        let state_strides = [h * dh * n, dh * n, n, 1u32];
        let group_size = (h / g) as i32;
        let state_size = n as i32;

        self.ssd_update.encode(
            conv_x,
            time_step,
            state_b,
            state_c,
            &self.skip_connection_weight,
            gate,
            None::<&Allocation<B>>,
            &mut out,
            &mut layer.ssm_state,
            group_size as u32,
            state_size as u32,
            &x_strides,
            &dt_strides,
            &cb_strides,
            &state_strides,
            suffix_length as u32,
            h,
            dh,
            encoder,
        );
        Ok(out)
    }

    pub fn encode(
        &self,
        args: MambaArguments<B>,
        input: Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let MambaArguments {
            active_row_count,
            layer,
        } = args;
        assert!(active_row_count > 0, "Mamba mixer requires at least one active row");

        let in_proj = self.in_projection.encode(input, active_row_count, encoder)?;
        let SplitInProjectionOutput {
            conv_inputs,
            gate,
            time_step,
        } = self.run_split_inproj(&in_proj, active_row_count, encoder)?;
        let ConvScanOutput {
            conv_x,
            state_b,
            state_c,
        } = self.run_conv_scan(layer, &conv_inputs, encoder, active_row_count)?;
        if active_row_count == 1 {
            let ssm_output =
                self.run_decode_ssm(layer, &conv_x, &state_b, &state_c, &time_step, &gate, encoder, active_row_count)?;
            self.out_projection.encode(ssm_output, active_row_count, encoder)
        } else {
            let ssm_output =
                self.run_prefill_ssm(layer, &conv_x, &state_b, &state_c, &time_step, &gate, encoder, active_row_count)?;
            self.out_projection.encode(ssm_output, active_row_count, encoder)
        }
    }
}
