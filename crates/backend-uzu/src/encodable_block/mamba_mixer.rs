//! Mamba2 SSM mixer encodable.

use std::env;

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        kernel::{
            Conv1dDecodeKernel, Conv1dPackKernel, Conv1dScanKernel, SSDUpdateKernel, SplitInProjKernel,
            ssd_prefill::{SSDPrefillArguments, SSDPrefillKernels, SSDPrefillMode},
        },
    },
    config::{DecoderLayerType, Mamba2Config},
    encodable_block::linear::Linear,
    forward_pass::ssm_layer::SSMLayer,
    parameters::{ParameterTree, resolve_subtree},
};

pub(crate) struct MambaMixer<B: Backend> {
    config: Mamba2Config,
    in_projection: Box<dyn Linear<B>>,
    out_projection: Box<dyn Linear<B>>,
    split_inproj: <B::Kernels as Kernels>::SplitInProjKernel,
    conv_decode: <B::Kernels as Kernels>::Conv1dDecodeKernel,
    conv_pack: <B::Kernels as Kernels>::Conv1dPackKernel,
    conv_scan: <B::Kernels as Kernels>::Conv1dScanKernel,
    ssd_prefill: SSDPrefillKernels<B>,
    ssd_update: <B::Kernels as Kernels>::SSDUpdateKernel,
    conv_weight: Allocation<B>,
    conv_bias: Option<Allocation<B>>,
    gate_bias: Allocation<B>,
    skip_connection_weight: Allocation<B>,
    prefill_mode: SSDPrefillMode,
    data_type: DataType,
}

pub(crate) struct MambaArguments<'a, B: Backend> {
    pub active_row_count: usize,
    pub layer: &'a mut SSMLayer<B>,
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        context: &B::Context,
        layer_type: DecoderLayerType,
        mamba_config: Mamba2Config,
        layer_index: usize,
        model_dim: usize,
        decoder_layer_loader: &ParameterTree<B::Context>,
    ) -> (Self, Option<Allocation<B>>) {
        if !matches!(layer_type, DecoderLayerType::StateSpace { .. }) {
            panic!("Layer {} marked as transformer but Mamba mixer config provided", layer_index);
        }
        let split_tree = resolve_subtree(decoder_layer_loader, &["mixer"]);
        let conv_tree = resolve_subtree(&split_tree, &["conv", "conv1d"]);

        let data_type: DataType = mamba_config.in_projection_config.activation_precision().into();

        let (in_projection, in_projection_input_hadamard_factors) = <dyn Linear<B>>::new_extracting_input_hadamard(
            &mamba_config.in_projection_config,
            model_dim,
            [mamba_config.conv_dim(), mamba_config.inner_dim(), mamba_config.num_heads],
            context,
            &resolve_subtree(decoder_layer_loader, &["mixer.in_projection", "mixer.in_proj"]),
        )
        .expect("Failed to create in-projection kernel");

        let out_projection = <dyn Linear<B>>::new(
            &mamba_config.out_projection_config,
            mamba_config.inner_dim(),
            [model_dim],
            context,
            &resolve_subtree(decoder_layer_loader, &["mixer.out_projection", "mixer.out_proj"]),
        )
        .expect("Failed to create out-projection kernel");

        let conv_weight = conv_tree.leaf("weights").unwrap().read_allocation().unwrap();
        let conv_bias = if mamba_config.conv_config.has_biases {
            Some(conv_tree.leaf("biases").unwrap().read_allocation().unwrap())
        } else {
            None
        };
        let gate_bias = split_tree.leaf("gate_bias").unwrap().read_allocation().unwrap();
        let skip_connection_weight = split_tree.leaf("skip_connection_weight").unwrap().read_allocation().unwrap();

        let split_inproj = <B::Kernels as Kernels>::SplitInProjKernel::new(context, data_type)
            .expect("Failed to create split in-projection kernel");
        let conv_scan =
            <B::Kernels as Kernels>::Conv1dScanKernel::new(context, data_type, mamba_config.conv_config.has_biases)
                .expect("Failed to create conv scan kernel");
        let conv_decode = <B::Kernels as Kernels>::Conv1dDecodeKernel::new(
            context,
            data_type,
            mamba_config.conv_config.has_biases,
            true,
        )
        .expect("Failed to create conv decode kernel");
        let conv_pack = <B::Kernels as Kernels>::Conv1dPackKernel::new(context, data_type)
            .expect("Failed to create conv pack kernel");
        let ssd_prefill = SSDPrefillKernels::new(context, data_type).expect("Failed to create SSD prefill kernel");
        let ssd_update = <B::Kernels as Kernels>::SSDUpdateKernel::new(context, data_type, true)
            .expect("Failed to create SSD decode kernel");
        let prefill_mode = resolve_prefill_mode_from_env();

        (
            Self {
                config: mamba_config,
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
                data_type,
            },
            in_projection_input_hadamard_factors,
        )
    }

    fn run_split_inproj(
        &self,
        in_proj: &Allocation<B>,
        conv_inputs: &mut Allocation<B>,
        gate: &mut Allocation<B>,
        dt: &mut Allocation<B>,
        suffix_length: usize,
        encoder: &mut Encoder<B>,
    ) {
        let conv_dim = self.config.conv_dim();
        let inner_dim = self.config.inner_dim();
        let num_heads = self.config.num_heads;
        let total_dim = conv_dim + inner_dim + num_heads;
        self.split_inproj.encode(
            in_proj,
            conv_inputs,
            gate,
            dt,
            &self.gate_bias,
            suffix_length as u32,
            total_dim as u32,
            conv_dim as u32,
            inner_dim as u32,
            num_heads as u32,
            encoder,
        );
    }

    fn run_conv_scan(
        &self,
        layer: &mut SSMLayer<B>,
        conv_inputs: &Allocation<B>,
        x: &mut Allocation<B>,
        b: &mut Allocation<B>,
        c: &mut Allocation<B>,
        mut padded: Option<&mut Allocation<B>>,
        encoder: &mut Encoder<B>,
        suffix_length: usize,
    ) -> Result<(), B::Error> {
        let conv_dim = self.config.conv_dim();
        let inner_dim = self.config.inner_dim();
        let proj_dim = self.config.num_groups * self.config.state_dim;
        let state_stride = self.config.kernel_size.saturating_sub(1);

        if suffix_length == 1 {
            if conv_dim > 0 && self.config.kernel_size > 0 {
                let next_state = layer.conv_state.as_mut().unwrap_or(&mut layer.ssm_state);
                self.conv_decode.encode(
                    conv_inputs,
                    &self.conv_weight,
                    self.conv_bias.as_ref(),
                    None::<&Allocation<B>>,
                    x,
                    b,
                    c,
                    next_state,
                    self.config.kernel_size as u32,
                    conv_dim as u32,
                    state_stride as u32,
                    conv_dim as u32,
                    suffix_length as u32,
                    inner_dim as u32,
                    proj_dim as u32,
                    self.config.activation.act_type(),
                    encoder,
                );
            }
        } else {
            if let Some(padded) = padded.as_mut() {
                let state_in = layer.conv_state.as_ref().unwrap_or(conv_inputs);
                self.conv_pack.encode(
                    state_in,
                    conv_inputs,
                    &mut **padded,
                    state_stride as u32,
                    conv_dim as u32,
                    suffix_length as u32,
                    conv_dim as u32,
                    encoder,
                );
            }

            if conv_dim > 0 && self.config.kernel_size > 0 {
                let conv_source = padded.as_deref().unwrap_or(conv_inputs);
                let next_state = layer.conv_state.as_mut().unwrap_or(&mut layer.ssm_state);
                self.conv_scan.encode(
                    conv_source,
                    &self.conv_weight,
                    self.conv_bias.as_ref(),
                    x,
                    b,
                    c,
                    next_state,
                    suffix_length as u32,
                    self.config.kernel_size as u32,
                    conv_dim as u32,
                    state_stride as u32,
                    conv_dim as u32,
                    inner_dim as u32,
                    proj_dim as u32,
                    self.config.activation.act_type(),
                    encoder,
                );
            }
        }
        Ok(())
    }

    fn run_prefill_ssm(
        &self,
        layer: &mut SSMLayer<B>,
        x: &Allocation<B>,
        b: &Allocation<B>,
        c: &Allocation<B>,
        dt: &Allocation<B>,
        z: &Allocation<B>,
        out: &mut Allocation<B>,
        encoder: &mut Encoder<B>,
        suffix_length: usize,
    ) {
        self.ssd_prefill.encode(
            encoder,
            SSDPrefillArguments {
                x,
                dt,
                b,
                c,
                d: &self.skip_connection_weight,
                z,
                state: &mut layer.ssm_state,
                y: out,
                suffix_len: suffix_length,
                group_size: (self.config.num_heads / self.config.num_groups) as u32,
                state_size: self.config.state_dim as u32,
                x_strides: [self.config.num_heads * self.config.head_dim, self.config.head_dim, 1],
                dt_strides: [self.config.num_heads, 1],
                cb_strides: [self.config.num_groups * self.config.state_dim, self.config.state_dim, 1],
                state_strides: [self.config.head_dim * self.config.state_dim, self.config.state_dim, 1],
                channels: self.config.num_heads,
                head_dim: self.config.head_dim,
            },
            self.prefill_mode,
        )
    }

    fn run_decode_ssm(
        &self,
        layer: &mut SSMLayer<B>,
        x: &Allocation<B>,
        b: &Allocation<B>,
        c: &Allocation<B>,
        dt: &Allocation<B>,
        z: &Allocation<B>,
        out: &mut Allocation<B>,
        encoder: &mut Encoder<B>,
        suffix_length: usize,
    ) {
        let h = self.config.num_heads as u32;
        let g = self.config.num_groups as u32;
        let dh = self.config.head_dim as u32;
        let n = self.config.state_dim as u32;

        let x_strides = [h * dh, dh, 1u32];
        let dt_strides = [h, 1u32];
        let cb_strides = [g * n, n, 1u32];
        let state_strides = [h * dh * n, dh * n, n, 1u32];
        let group_size = (h / g) as i32;
        let state_size = n as i32;

        self.ssd_update.encode(
            x,
            dt,
            b,
            c,
            &self.skip_connection_weight,
            z,
            None::<&Allocation<B>>,
            out,
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
    }

    pub(crate) fn encode(
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
        let mut conv_inputs =
            encoder.allocate_scratch(size_for_shape(&[active_row_count, self.config.conv_dim()], self.data_type))?;
        let mut z = encoder.allocate_scratch(size_for_shape(
            &[active_row_count, self.config.num_heads, self.config.head_dim],
            self.data_type,
        ))?;
        let mut dt =
            encoder.allocate_scratch(size_for_shape(&[active_row_count, self.config.num_heads], self.data_type))?;
        self.run_split_inproj(&in_proj, &mut conv_inputs, &mut z, &mut dt, active_row_count, encoder);

        let mut x = encoder.allocate_scratch(size_for_shape(
            &[active_row_count, self.config.num_heads, self.config.head_dim],
            self.data_type,
        ))?;
        let bc_shape = [active_row_count, self.config.num_groups, self.config.state_dim];
        let mut b = encoder.allocate_scratch(size_for_shape(&bc_shape, self.data_type))?;
        let mut c = encoder.allocate_scratch(size_for_shape(&bc_shape, self.data_type))?;
        let state_stride = self.config.kernel_size.saturating_sub(1);
        let mut padded = (active_row_count > 1 && state_stride > 0)
            .then(|| {
                encoder.allocate_scratch(size_for_shape(
                    &[active_row_count + state_stride, self.config.conv_dim()],
                    self.data_type,
                ))
            })
            .transpose()?;
        self.run_conv_scan(layer, &conv_inputs, &mut x, &mut b, &mut c, padded.as_mut(), encoder, active_row_count)?;

        let mut ssm_output =
            encoder.allocate_scratch(size_for_shape(&[active_row_count, self.config.inner_dim()], self.data_type))?;
        if active_row_count == 1 {
            self.run_decode_ssm(layer, &x, &b, &c, &dt, &z, &mut ssm_output, encoder, active_row_count);
        } else {
            self.run_prefill_ssm(layer, &x, &b, &c, &dt, &z, &mut ssm_output, encoder, active_row_count);
        }

        self.out_projection.encode(ssm_output, active_row_count, encoder)
    }
}
