//! Mamba2 SSM mixer encodable.

use std::env;

use metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};
use objc2::{Message, rc::Retained, runtime::ProtocolObject};

use super::transformer_layer;
use crate::{
    Activation, DataType,
    array::Array,
    backends::{
        common::kernel::{Conv1dDecodeKernel, Conv1dPackKernel, Conv1dScanKernel, SSDUpdateKernel, SplitInProjKernel},
        metal::{
            Metal, MetalContext,
            kernel::{
                dsl::{
                    Conv1dDecodeMetalKernel, Conv1dPackMetalKernel, Conv1dScanMetalKernel, SSDUpdateMetalKernel,
                    SplitInProjMetalKernel,
                },
                ssm::ssd_prefill::{SSDPrefillArguments, SSDPrefillKernels, SSDPrefillMode},
            },
        },
    },
    config::{DecoderLayerType, Mamba2Config},
    encodable_block::{EncodableBlock, EncodingParameters},
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::ParameterTree,
};

pub(crate) struct MambaMixer {
    layer_index: usize,
    config: Mamba2Config,
    in_projection: Box<dyn EncodableBlock<Metal>>,
    out_projection: Box<dyn EncodableBlock<Metal>>,
    split_inproj: SplitInProjMetalKernel,
    conv_decode: Conv1dDecodeMetalKernel,
    conv_pack: Conv1dPackMetalKernel,
    conv_scan: Conv1dScanMetalKernel,
    ssd_prefill: SSDPrefillKernels,
    ssd_update: SSDUpdateMetalKernel,
    conv_weight: Array<Metal>,
    conv_bias: Option<Array<Metal>>,
    gate_bias: Array<Metal>,
    skip_connection_weight: Array<Metal>,
    prefill_mode: SSDPrefillMode,
}

impl MambaMixer {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        mtl_context: &MetalContext,
        layer_type: DecoderLayerType,
        mamba_config: Mamba2Config,
        layer_index: usize,
        model_dim: usize,
        num_heads: usize,
        head_dim: usize,
        num_groups: usize,
        decoder_layer_loader: &ParameterTree<MetalContext>,
    ) -> Self {
        let _ = (num_heads, head_dim, num_groups);
        if !matches!(layer_type, DecoderLayerType::StateSpace { .. }) {
            panic!("Layer {} marked as transformer but Mamba mixer config provided", layer_index);
        }
        let split_tree = resolve_subtree(decoder_layer_loader, &["mixer"]);
        let conv_tree = resolve_subtree(&split_tree, &["conv", "conv1d"]);

        let data_type: DataType = mamba_config.in_projection_config.activation_precision().into();

        let in_projection = transformer_layer::linear_block(
            &mamba_config.in_projection_config,
            mamba_config.has_in_biases,
            model_dim,
            [mamba_config.conv_dim(), mamba_config.inner_dim(), mamba_config.num_heads],
            mtl_context,
            &resolve_subtree(decoder_layer_loader, &["mixer.in_projection", "mixer.in_proj"]),
            ArrayId::Main,
            ArrayId::SsmInProj,
        )
        .expect("Failed to create in-projection kernel");

        let out_projection = transformer_layer::linear_block(
            &mamba_config.out_projection_config,
            mamba_config.has_out_biases,
            mamba_config.inner_dim(),
            [model_dim],
            mtl_context,
            &resolve_subtree(decoder_layer_loader, &["mixer.out_projection", "mixer.out_proj"]),
            ArrayId::AttentionOutput,
            ArrayId::Main,
        )
        .expect("Failed to create out-projection kernel");

        let conv_weight = conv_tree.leaf("weights").unwrap().clone();
        let conv_bias = if mamba_config.conv_config.has_biases {
            Some(conv_tree.leaf("biases").unwrap().clone())
        } else {
            None
        };
        let gate_bias = split_tree.leaf("gate_bias").unwrap().clone();
        let skip_connection_weight = split_tree.leaf("skip_connection_weight").unwrap().clone();

        let split_inproj =
            SplitInProjMetalKernel::new(mtl_context, data_type).expect("Failed to create split in-projection kernel");
        let conv_scan = Conv1dScanMetalKernel::new(
            mtl_context,
            data_type,
            Self::activation_to_int(&mamba_config.activation),
            mamba_config.conv_config.has_biases,
        )
        .expect("Failed to create conv scan kernel");
        let conv_decode = Conv1dDecodeMetalKernel::new(
            mtl_context,
            data_type,
            Self::activation_to_int(&mamba_config.activation),
            mamba_config.conv_config.has_biases,
        )
        .expect("Failed to create conv decode kernel");
        let conv_pack = Conv1dPackMetalKernel::new(mtl_context, data_type).expect("Failed to create conv pack kernel");
        let ssd_prefill = SSDPrefillKernels::new(mtl_context, data_type).expect("Failed to create SSD prefill kernel");
        let ssd_update = SSDUpdateMetalKernel::new(mtl_context, data_type).expect("Failed to create SSD decode kernel");
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

    fn encode_pipeline_with_encoder(
        &self,
        state: &mut ForwardPassState<Metal>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        parameters: &EncodingParameters<Metal>,
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
        state: &mut ForwardPassState<Metal>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        suffix_length: usize,
    ) {
        let arrays = state.arrays(&[
            ArrayId::SsmInProj,
            ArrayId::SsmPacked(self.layer_index),
            ArrayId::SsmZ(self.layer_index),
            ArrayId::SsmDt(self.layer_index),
        ]);
        let in_proj = arrays[0].borrow_mut();
        let conv_inputs = arrays[1].borrow_mut();
        let gate = arrays[2].borrow_mut();
        let dt = arrays[3].borrow_mut();

        let input_buf = in_proj.buffer().to_owned();
        let conv_buf = conv_inputs.buffer().to_owned();
        let gate_buf = gate.buffer().to_owned();
        let dt_buf = dt.buffer().to_owned();
        let gate_bias = self.gate_bias.clone();
        let bias_buf = gate_bias.buffer().to_owned();

        let conv_dim = self.config.conv_dim();
        let inner_dim = self.config.inner_dim();
        let num_heads = self.config.num_heads;
        let total_dim = conv_dim + inner_dim + num_heads;

        self.split_inproj.encode(
            &input_buf,
            &conv_buf,
            &gate_buf,
            &dt_buf,
            &bias_buf,
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
        state: &mut ForwardPassState<Metal>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        suffix_length: usize,
    ) {
        let arrays = state.arrays(&[
            ArrayId::SsmPacked(self.layer_index),
            ArrayId::SsmConvState(self.layer_index),
            ArrayId::SsmX(self.layer_index),
            ArrayId::SsmB(self.layer_index),
            ArrayId::SsmC(self.layer_index),
        ]);
        let conv_inputs = arrays[0].borrow_mut();
        let conv_state = arrays[1].borrow_mut();
        let x_arr = arrays[2].borrow_mut();
        let b_arr = arrays[3].borrow_mut();
        let c_arr = arrays[4].borrow_mut();

        let input_buf = conv_inputs.buffer().to_owned();
        let state_buf = conv_state.buffer().clone();
        let x_buf = x_arr.buffer().to_owned();
        let b_buf = b_arr.buffer().to_owned();
        let c_buf = c_arr.buffer().to_owned();

        let weight_storage = self.conv_weight.clone();
        let weight_buf = weight_storage.buffer().clone();
        let bias_buf = self.conv_bias.as_ref().map(|arr| arr.buffer());

        let conv_dim = self.config.conv_dim();
        let inner_dim = self.config.inner_dim();
        let proj_dim = self.config.num_groups * self.config.state_dim;
        let state_stride = self.config.kernel_size.saturating_sub(1);
        drop(conv_state);

        if suffix_length == 1 {
            if conv_dim > 0 && self.config.kernel_size > 0 {
                self.conv_decode.encode(
                    &input_buf,
                    &weight_buf,
                    bias_buf,
                    &state_buf,
                    &x_buf,
                    &b_buf,
                    &c_buf,
                    &state_buf,
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
                let borrow = array.borrow_mut();
                let buf = borrow.buffer().clone();
                drop(borrow);

                self.conv_pack.encode(
                    &state_buf,
                    &input_buf,
                    &buf,
                    state_stride as u32,
                    conv_dim as u32,
                    suffix_length as u32,
                    conv_dim as u32,
                    &encoder,
                );

                Some(buf)
            } else {
                None
            };

            if conv_dim > 0 && self.config.kernel_size > 0 {
                let conv_source = padded_buf.as_deref().unwrap_or(&input_buf).retain();
                self.conv_scan.encode(
                    &conv_source,
                    &weight_buf,
                    bias_buf,
                    &x_buf,
                    &b_buf,
                    &c_buf,
                    &state_buf,
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
        state: &mut ForwardPassState<Metal>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
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
        let x_buf = x.buffer().clone();
        drop(x);
        let b = base_arrays[1].borrow();
        let b_buf = b.buffer().clone();
        drop(b);
        let c = base_arrays[2].borrow();
        let c_buf = c.buffer().clone();
        drop(c);
        let dt = base_arrays[3].borrow();
        let dt_buf = dt.buffer().clone();
        drop(dt);
        let z = base_arrays[4].borrow();
        let z_buf = z.buffer().clone();
        drop(z);
        let state_buf = base_arrays[5].borrow();
        let state_raw = state_buf.buffer().clone();
        drop(state_buf);
        let out = base_arrays[6].borrow();
        let out_buf = out.buffer().clone();
        drop(out);

        let skip_weights = self.skip_connection_weight.clone();
        let skip = skip_weights.buffer().clone();

        self.ssd_prefill.encode(
            encoder,
            SSDPrefillArguments {
                x: &x_buf,
                dt: &dt_buf,
                b: &b_buf,
                c: &c_buf,
                d: &skip,
                z: &z_buf,
                state: &state_raw,
                y: &out_buf,
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
        state: &mut ForwardPassState<Metal>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
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
        let x_buf = x.buffer().clone();
        drop(x);
        let b = arrays[1].borrow();
        let b_buf = b.buffer().clone();
        drop(b);
        let c = arrays[2].borrow();
        let c_buf = c.buffer().clone();
        drop(c);
        let dt = arrays[3].borrow();
        let dt_buf = dt.buffer().clone();
        drop(dt);
        let z = arrays[4].borrow();
        let z_buf = z.buffer().clone();
        drop(z);
        let state_arr = arrays[5].borrow();
        let state_buf = state_arr.buffer().clone();
        drop(state_arr);
        let y = arrays[6].borrow();
        let y_buf = y.buffer().clone();
        let skip_weights = self.skip_connection_weight.clone();
        let skip_buf = skip_weights.buffer().clone();

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
            &x_buf,
            &dt_buf,
            &b_buf,
            &c_buf,
            &skip_buf,
            &z_buf,
            &state_buf,
            &y_buf,
            &state_buf,
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

impl EncodableBlock<Metal> for MambaMixer {
    fn encode(
        &self,
        state: &mut ForwardPassState<Metal>,
        parameters: &EncodingParameters<Metal>,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    ) {
        if self.supports_shared_encoder() {
            let encoder =
                command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
            self.encode_pipeline_with_encoder(state, &encoder, parameters);
            encoder.end_encoding();

            if parameters.wait_until_completed {
                command_buffer.commit();
                command_buffer.wait_until_completed();
            }
            return;
        }

        let active_suffix_length = state.active_suffix_length();
        if active_suffix_length == 0 {
            return;
        }

        self.in_projection.encode(state, parameters, command_buffer);

        let encoder = command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
        self.run_split_inproj(state, &encoder, active_suffix_length);
        self.run_conv_scan(state, &encoder, active_suffix_length);
        if active_suffix_length == 1 {
            self.run_decode_ssm(state, &encoder, active_suffix_length);
        } else {
            self.run_prefill_ssm(state, &encoder, active_suffix_length);
        }
        encoder.end_encoding();

        self.out_projection.encode(state, parameters, command_buffer);

        if parameters.wait_until_completed {
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        self.in_projection.supports_shared_encoder() && self.out_projection.supports_shared_encoder()
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<Metal>,
        parameters: &EncodingParameters<Metal>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    ) {
        self.encode_pipeline_with_encoder(state, encoder, parameters);
    }
}

fn resolve_subtree<'tree>(
    loader: &'tree ParameterTree<MetalContext>,
    candidates: &[&str],
) -> ParameterTree<'tree, MetalContext> {
    for candidate in candidates {
        if let Ok(tree) = loader.subtree(candidate) {
            return tree;
        }
    }

    let missing = candidates.first().copied().unwrap_or("<missing subtree name>");
    loader.subtree(missing).unwrap_or_else(|_| panic!("Unable to resolve parameter subtree '{missing}'"))
}
