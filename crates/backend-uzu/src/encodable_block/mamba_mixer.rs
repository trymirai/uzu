//! Mamba2 SSM mixer encodable.

use std::{
    env,
    ops::{Deref, DerefMut},
};

use crate::{
    DataType,
    array::Array,
    backends::common::{
        Backend, Encoder, Kernels,
        kernel::{
            Conv1dDecodeKernel, Conv1dPackKernel, Conv1dScanKernel, SSDUpdateKernel, SplitInProjKernel,
            ssd_prefill::{SSDPrefillArguments, SSDPrefillKernels, SSDPrefillMode},
        },
    },
    config::{DecoderLayerType, Mamba2Config},
    encodable_block::linear::Linear,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterTree, resolve_subtree},
};

pub(crate) struct MambaMixer<B: Backend> {
    layer_index: usize,
    config: Mamba2Config,
    in_projection: Box<dyn Linear<B>>,
    out_projection: Box<dyn Linear<B>>,
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

        let in_projection = <dyn Linear<B>>::new(
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

        let out_projection = <dyn Linear<B>>::new(
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

        let conv_weight = conv_tree.leaf_array("weights").unwrap();
        let conv_bias = if mamba_config.conv_config.has_biases {
            Some(conv_tree.leaf_array("biases").unwrap())
        } else {
            None
        };
        let gate_bias = split_tree.leaf_array("gate_bias").unwrap();
        let skip_connection_weight = split_tree.leaf_array("skip_connection_weight").unwrap();

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

    fn run_split_inproj(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
        suffix_length: usize,
    ) {
        let in_proj = state.array(ArrayId::SsmInProj);
        let conv_inputs = state.array(ArrayId::SsmPacked(self.layer_index));
        let gate = state.array(ArrayId::SsmZ(self.layer_index));
        let dt = state.array(ArrayId::SsmDt(self.layer_index));

        let conv_dim = self.config.conv_dim();
        let inner_dim = self.config.inner_dim();
        let num_heads = self.config.num_heads;
        let total_dim = conv_dim + inner_dim + num_heads;

        self.split_inproj.encode(
            in_proj.buffer().borrow().deref(),
            conv_inputs.buffer().borrow_mut().deref_mut(),
            gate.buffer().borrow_mut().deref_mut(),
            dt.buffer().borrow_mut().deref_mut(),
            self.gate_bias.buffer().borrow().deref(),
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
        encoder: &mut Encoder<B>,
        suffix_length: usize,
    ) {
        let conv_inputs = state.array(ArrayId::SsmPacked(self.layer_index));
        let conv_state = state.array(ArrayId::SsmConvState(self.layer_index));
        let x_arr = state.array(ArrayId::SsmX(self.layer_index));
        let b_arr = state.array(ArrayId::SsmB(self.layer_index));
        let c_arr = state.array(ArrayId::SsmC(self.layer_index));

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
                    None::<&B::Buffer>,
                    x_arr.buffer().borrow_mut().deref_mut(),
                    b_arr.buffer().borrow_mut().deref_mut(),
                    c_arr.buffer().borrow_mut().deref_mut(),
                    conv_state.buffer().borrow_mut().deref_mut(),
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
            let padded_buf = if state_stride > 0 {
                let array = state.conv_padded_buffer().expect("Missing conv padded buffer");
                let buffer = array.buffer();

                self.conv_pack.encode(
                    conv_state.buffer().borrow().deref(),
                    conv_inputs.buffer().borrow().deref(),
                    buffer.borrow_mut().deref_mut(),
                    state_stride as u32,
                    conv_dim as u32,
                    suffix_length as u32,
                    conv_dim as u32,
                    encoder,
                );

                Some(buffer)
            } else {
                None
            };

            if conv_dim > 0 && self.config.kernel_size > 0 {
                let padded_borrow = padded_buf.as_ref().map(|b| b.borrow());
                let conv_inputs_borrow = conv_inputs.buffer();
                let conv_inputs_ref = conv_inputs_borrow.borrow();
                let conv_source: &B::Buffer = padded_borrow.as_deref().unwrap_or(conv_inputs_ref.deref());
                self.conv_scan.encode(
                    conv_source,
                    weight_buf_borrow.deref(),
                    bias_buf_borrow.as_deref(),
                    x_arr.buffer().borrow_mut().deref_mut(),
                    b_arr.buffer().borrow_mut().deref_mut(),
                    c_arr.buffer().borrow_mut().deref_mut(),
                    conv_state.buffer().borrow_mut().deref_mut(),
                    suffix_length as u32,
                    self.config.kernel_size as u32,
                    conv_dim as u32,
                    state_stride as u32,
                    conv_dim as u32,
                    inner_dim as u32,
                    proj_dim as u32,
                    self.config.activation.act_type(),
                    encoder,
                )
            }
        }
    }

    fn run_prefill_ssm(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
        suffix_length: usize,
    ) {
        let x = state.array(ArrayId::SsmX(self.layer_index));
        let b = state.array(ArrayId::SsmB(self.layer_index));
        let c = state.array(ArrayId::SsmC(self.layer_index));
        let dt = state.array(ArrayId::SsmDt(self.layer_index));
        let z = state.array(ArrayId::SsmZ(self.layer_index));
        let state_arr = state.array(ArrayId::SsmState(self.layer_index));
        let out = state.array(ArrayId::AttentionOutput);

        self.ssd_prefill.encode(
            encoder,
            SSDPrefillArguments {
                x: x.buffer().borrow().deref(),
                dt: dt.buffer().borrow().deref(),
                b: b.buffer().borrow().deref(),
                c: c.buffer().borrow().deref(),
                d: self.skip_connection_weight.buffer().borrow().deref(),
                z: z.buffer().borrow().deref(),
                state: state_arr.buffer().borrow_mut().deref_mut(),
                y: out.buffer().borrow_mut().deref_mut(),
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
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
        suffix_length: usize,
    ) {
        let x = state.array(ArrayId::SsmX(self.layer_index));
        let b = state.array(ArrayId::SsmB(self.layer_index));
        let c = state.array(ArrayId::SsmC(self.layer_index));
        let dt = state.array(ArrayId::SsmDt(self.layer_index));
        let z = state.array(ArrayId::SsmZ(self.layer_index));
        let state_arr = state.array(ArrayId::SsmState(self.layer_index));
        let y = state.array(ArrayId::AttentionOutput);

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
            self.skip_connection_weight.buffer().borrow().deref(),
            z.buffer().borrow().deref(),
            None::<&B::Buffer>,
            y.buffer().borrow_mut().deref_mut(),
            state_arr.buffer().borrow_mut().deref_mut(),
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
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let active_row_count = state.active_row_count();
        if active_row_count == 0 {
            return Ok(());
        }

        self.in_projection.encode(state, encoder)?;
        self.run_split_inproj(state, encoder, active_row_count);
        self.run_conv_scan(state, encoder, active_row_count);
        if active_row_count == 1 {
            self.run_decode_ssm(state, encoder, active_row_count);
        } else {
            self.run_prefill_ssm(state, encoder, active_row_count);
        }
        self.out_projection.encode(state, encoder)?;
        Ok(())
    }
}
