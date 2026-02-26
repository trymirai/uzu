//! Mamba2 SSM mixer encodable.

use std::{env, ops::Deref};

use crate::{
    Activation, DataType,
    array::Array,
    backends::common::{
        Backend, CommandBuffer, Kernels,
        kernel::{
            Conv1dDecodeKernel, Conv1dPackKernel, Conv1dScanKernel, SSDUpdateKernel, SplitInProjKernel,
            matmul::MatmulKernels,
            ssd_prefill::{SSDPrefillArguments, SSDPrefillKernels, SSDPrefillMode},
        },
    },
    config::{DecoderLayerType, Mamba2Config},
    encodable_block::{EncodableBlock, EncodingParameters, transformer_layer::linear_block},
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterTree, resolve_subtree},
};

pub(crate) struct MambaMixer<B: Backend> {
    layer_index: usize,
    config: Mamba2Config,
    in_projection: Box<dyn EncodableBlock<B>>,
    out_projection: Box<dyn EncodableBlock<B>>,
    split_inproj: <B::Kernels as Kernels>::SplitInProjKernel,
    conv_decode: <B::Kernels as Kernels>::Conv1dDecodeKernel,
    conv_pack: <B::Kernels as Kernels>::Conv1dPackKernel,
    conv_scan: <B::Kernels as Kernels>::Conv1dScanKernel,
    ssd_prefill: SSDPrefillKernels<B>,
    ssd_update: <B::Kernels as Kernels>::SSDUpdateKernel,
    conv_weight: Array<B>,
    conv_bias: Option<Array<B>>,
    gate_bias: Array<B>,
    skip_connection_weight: Array<B>,
    prefill_mode: SSDPrefillMode,
}

impl<B: Backend + 'static> MambaMixer<B>
where
    B::Kernels: MatmulKernels,
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        context: &B::Context,
        layer_type: DecoderLayerType,
        mamba_config: Mamba2Config,
        layer_index: usize,
        model_dim: usize,
        num_heads: usize,
        head_dim: usize,
        num_groups: usize,
        decoder_layer_loader: &ParameterTree<B::Context>,
    ) -> Self {
        let _ = (num_heads, head_dim, num_groups);
        if !matches!(layer_type, DecoderLayerType::StateSpace { .. }) {
            panic!("Layer {} marked as transformer but Mamba mixer config provided", layer_index);
        }
        let split_tree = resolve_subtree(decoder_layer_loader, &["mixer"]);
        let conv_tree = resolve_subtree(&split_tree, &["conv", "conv1d"]);

        let data_type: DataType = mamba_config.in_projection_config.activation_precision().into();

        let in_projection = linear_block(
            &mamba_config.in_projection_config,
            mamba_config.has_in_biases,
            model_dim,
            [mamba_config.conv_dim(), mamba_config.inner_dim(), mamba_config.num_heads],
            context,
            &resolve_subtree(decoder_layer_loader, &["mixer.in_projection", "mixer.in_proj"]),
            ArrayId::Main,
            ArrayId::SsmInProj,
        )
        .expect("Failed to create in-projection kernel");

        let out_projection = linear_block(
            &mamba_config.out_projection_config,
            mamba_config.has_out_biases,
            mamba_config.inner_dim(),
            [model_dim],
            context,
            &resolve_subtree(decoder_layer_loader, &["mixer.out_projection", "mixer.out_proj"]),
            ArrayId::AttentionOutput,
            ArrayId::Main,
        )
        .expect("Failed to create out-projection kernel");

        let conv_weight = conv_tree.leaf("weights").unwrap();
        let conv_bias = if mamba_config.conv_config.has_biases {
            Some(conv_tree.leaf("biases").unwrap())
        } else {
            None
        };
        let gate_bias = split_tree.leaf("gate_bias").unwrap();
        let skip_connection_weight = split_tree.leaf("skip_connection_weight").unwrap();

        let split_inproj = <B::Kernels as Kernels>::SplitInProjKernel::new(context, data_type)
            .expect("Failed to create split in-projection kernel");
        let conv_scan = <B::Kernels as Kernels>::Conv1dScanKernel::new(
            context,
            data_type,
            Self::activation_to_int(&mamba_config.activation),
            mamba_config.conv_config.has_biases,
        )
        .expect("Failed to create conv scan kernel");
        let conv_decode = <B::Kernels as Kernels>::Conv1dDecodeKernel::new(
            context,
            data_type,
            Self::activation_to_int(&mamba_config.activation),
            mamba_config.conv_config.has_biases,
        )
        .expect("Failed to create conv decode kernel");
        let conv_pack = <B::Kernels as Kernels>::Conv1dPackKernel::new(context, data_type)
            .expect("Failed to create conv pack kernel");
        let ssd_prefill = SSDPrefillKernels::new(context, data_type).expect("Failed to create SSD prefill kernel");
        let ssd_update = <B::Kernels as Kernels>::SSDUpdateKernel::new(context, data_type)
            .expect("Failed to create SSD decode kernel");
        let prefill_mode = resolve_prefill_mode_from_env();

        Self {
            layer_index,
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
        }
    }
}

impl<B: Backend> MambaMixer<B> {
    fn encode_pipeline_with_encoder(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &B::ComputeEncoder,
        parameters: &EncodingParameters<B>,
    ) {
        let active_suffix_length = state.active_suffix_length();
        if active_suffix_length == 0 {
            return;
        }

        self.in_projection.encode_with_shared_encoder(state, parameters, encoder);
        self.run_split_inproj(state, encoder, active_suffix_length);
        self.run_conv_scan(state, encoder, active_suffix_length);

        if active_suffix_length == 1 {
            self.run_decode_ssm(state, encoder, active_suffix_length);
        } else {
            self.run_prefill_ssm(state, encoder, active_suffix_length);
        }

        self.out_projection.encode_with_shared_encoder(state, parameters, encoder);
    }

    fn run_split_inproj(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &B::ComputeEncoder,
        suffix_length: usize,
    ) {
        let arrays = state.arrays(&[
            ArrayId::SsmInProj,
            ArrayId::SsmPacked(self.layer_index),
            ArrayId::SsmZ(self.layer_index),
            ArrayId::SsmDt(self.layer_index),
        ]);

        let in_proj = arrays[0].borrow();
        let conv_inputs = arrays[1].borrow();
        let gate = arrays[2].borrow();
        let dt = arrays[3].borrow();

        let bias_buf_rc = self.gate_bias.buffer();
        let bias_buf_borrow = bias_buf_rc.borrow();
        let conv_dim = self.config.conv_dim();
        let inner_dim = self.config.inner_dim();
        let num_heads = self.config.num_heads;
        let total_dim = conv_dim + inner_dim + num_heads;

        self.split_inproj.encode(
            in_proj.buffer().borrow().deref(),
            conv_inputs.buffer().borrow().deref(),
            gate.buffer().borrow().deref(),
            dt.buffer().borrow().deref(),
            bias_buf_borrow.deref(),
            suffix_length as u32,
            total_dim as u32,
            conv_dim as u32,
            inner_dim as u32,
            num_heads as u32,
            encoder,
        )
    }

    fn run_conv_scan(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &B::ComputeEncoder,
        suffix_length: usize,
    ) {
        let arrays = state.arrays(&[
            ArrayId::SsmPacked(self.layer_index),
            ArrayId::SsmConvState(self.layer_index),
            ArrayId::SsmX(self.layer_index),
            ArrayId::SsmB(self.layer_index),
            ArrayId::SsmC(self.layer_index),
        ]);
        let conv_inputs = arrays[0].borrow();
        let conv_state = arrays[1].borrow();
        let x_arr = arrays[2].borrow();
        let b_arr = arrays[3].borrow();
        let c_arr = arrays[4].borrow();

        let weight_buf_rc = self.conv_weight.buffer();
        let weight_buf_borrow = weight_buf_rc.borrow();
        let bias_buf_rc = self.conv_bias.as_ref().map(|arr| arr.buffer());
        let bias_buf_borrow = bias_buf_rc.as_ref().map(|rc| rc.borrow());

        let conv_dim = self.config.conv_dim();
        let inner_dim = self.config.inner_dim();
        let proj_dim = self.config.num_groups * self.config.state_dim;
        let state_stride = self.config.kernel_size.saturating_sub(1);

        if suffix_length == 1 {
            if conv_dim > 0 && self.config.kernel_size > 0 {
                self.conv_decode.encode(
                    conv_inputs.buffer().borrow().deref(),
                    weight_buf_borrow.deref(),
                    bias_buf_borrow.as_deref(),
                    conv_state.buffer().borrow().deref(),
                    x_arr.buffer().borrow().deref(),
                    b_arr.buffer().borrow().deref(),
                    c_arr.buffer().borrow().deref(),
                    conv_state.buffer().borrow().deref(),
                    self.config.kernel_size as u32,
                    conv_dim as u32,
                    state_stride as u32,
                    conv_dim as u32,
                    suffix_length as u32,
                    inner_dim as u32,
                    proj_dim as u32,
                    &encoder,
                );
            }
        } else {
            let padded_buf = if state_stride > 0 {
                let array = state.conv_padded_buffer().expect("Missing conv padded buffer");
                let buffer = array.borrow().buffer();

                self.conv_pack.encode(
                    conv_state.buffer().borrow().deref(),
                    conv_inputs.buffer().borrow().deref(),
                    buffer.borrow().deref(),
                    state_stride as u32,
                    conv_dim as u32,
                    suffix_length as u32,
                    conv_dim as u32,
                    &encoder,
                );

                Some(buffer)
            } else {
                None
            };

            if conv_dim > 0 && self.config.kernel_size > 0 {
                let padded_borrow = padded_buf.as_ref().map(|b| b.borrow());
                let conv_inputs_borrow = conv_inputs.buffer();
                let conv_inputs_ref = conv_inputs_borrow.borrow();
                let conv_source: &B::NativeBuffer = padded_borrow.as_deref().unwrap_or(conv_inputs_ref.deref());
                self.conv_scan.encode(
                    conv_source,
                    weight_buf_borrow.deref(),
                    bias_buf_borrow.as_deref(),
                    x_arr.buffer().borrow().deref(),
                    b_arr.buffer().borrow().deref(),
                    c_arr.buffer().borrow().deref(),
                    conv_state.buffer().borrow().deref(),
                    suffix_length as u32,
                    self.config.kernel_size as u32,
                    conv_dim as u32,
                    state_stride as u32,
                    conv_dim as u32,
                    inner_dim as u32,
                    proj_dim as u32,
                    &encoder,
                )
            }
        }
    }

    fn run_prefill_ssm(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &B::ComputeEncoder,
        suffix_length: usize,
    ) {
        let base_arrays = state.arrays(&[
            ArrayId::SsmX(self.layer_index),
            ArrayId::SsmB(self.layer_index),
            ArrayId::SsmC(self.layer_index),
            ArrayId::SsmDt(self.layer_index),
            ArrayId::SsmZ(self.layer_index),
            ArrayId::SsmState(self.layer_index),
            ArrayId::AttentionOutput,
        ]);

        let x = base_arrays[0].borrow();
        let b = base_arrays[1].borrow();
        let c = base_arrays[2].borrow();
        let dt = base_arrays[3].borrow();
        let z = base_arrays[4].borrow();
        let state_arr = base_arrays[5].borrow();
        let out = base_arrays[6].borrow();

        let skip_rc = self.skip_connection_weight.buffer();
        let skip_borrow = skip_rc.borrow();
        self.ssd_prefill.encode(
            encoder,
            SSDPrefillArguments {
                x: x.buffer().borrow().deref(),
                dt: dt.buffer().borrow().deref(),
                b: b.buffer().borrow().deref(),
                c: c.buffer().borrow().deref(),
                d: skip_borrow.deref(),
                z: z.buffer().borrow().deref(),
                state: state_arr.buffer().borrow().deref(),
                y: out.buffer().borrow().deref(),
                suffix_len: suffix_length,
                group_size: (self.config.num_heads / self.config.num_groups) as i32,
                state_size: self.config.state_dim as i32,
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
        state: &mut ForwardPassState<B>,
        encoder: &B::ComputeEncoder,
        suffix_length: usize,
    ) {
        let arrays = state.arrays(&[
            ArrayId::SsmX(self.layer_index),
            ArrayId::SsmB(self.layer_index),
            ArrayId::SsmC(self.layer_index),
            ArrayId::SsmDt(self.layer_index),
            ArrayId::SsmZ(self.layer_index),
            ArrayId::SsmState(self.layer_index),
            ArrayId::AttentionOutput,
        ]);
        let x = arrays[0].borrow();
        let b = arrays[1].borrow();
        let c = arrays[2].borrow();
        let dt = arrays[3].borrow();
        let z = arrays[4].borrow();
        let state_arr = arrays[5].borrow();
        let y = arrays[6].borrow();

        let skip_rc = self.skip_connection_weight.buffer();
        let skip_borrow = skip_rc.borrow();

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
            x.buffer().borrow().deref(),
            dt.buffer().borrow().deref(),
            b.buffer().borrow().deref(),
            c.buffer().borrow().deref(),
            skip_borrow.deref(),
            z.buffer().borrow().deref(),
            state_arr.buffer().borrow().deref(),
            y.buffer().borrow().deref(),
            state_arr.buffer().borrow().deref(),
            group_size as u32,
            state_size as u32,
            x_strides.as_slice(),
            dt_strides.as_slice(),
            cb_strides.as_slice(),
            state_strides.as_slice(),
            suffix_length as u32,
            h,
            dh,
            encoder,
        );
    }

    fn activation_to_int(activation: &Activation) -> u32 {
        match activation {
            Activation::Identity => 0,
            Activation::SiLU {
                ..
            } => 1,
            Activation::Gelu => 2,
        }
    }
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

impl<B: Backend> EncodableBlock<B> for MambaMixer<B> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters<B>,
        command_buffer: &mut B::CommandBuffer,
    ) {
        if self.supports_shared_encoder() {
            command_buffer
                .with_compute_encoder(|encoder| self.encode_pipeline_with_encoder(state, encoder, parameters));

            if parameters.wait_until_completed {
                command_buffer.submit();
                command_buffer.wait_until_completed();
            }
            return;
        }

        let active_suffix_length = state.active_suffix_length();
        if active_suffix_length == 0 {
            return;
        }

        self.in_projection.encode(state, parameters, command_buffer);

        command_buffer.with_compute_encoder(|encoder| {
            self.run_split_inproj(state, encoder, active_suffix_length);
            self.run_conv_scan(state, encoder, active_suffix_length);
            if active_suffix_length == 1 {
                self.run_decode_ssm(state, encoder, active_suffix_length);
            } else {
                self.run_prefill_ssm(state, encoder, active_suffix_length);
            }
        });

        self.out_projection.encode(state, parameters, command_buffer);

        if parameters.wait_until_completed {
            command_buffer.submit();
            command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        self.in_projection.supports_shared_encoder() && self.out_projection.supports_shared_encoder()
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters<B>,
        encoder: &B::ComputeEncoder,
    ) {
        self.encode_pipeline_with_encoder(state, encoder, parameters);
    }
}
