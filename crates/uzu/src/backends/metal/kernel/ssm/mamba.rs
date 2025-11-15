use std::rc::Rc;

use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{
    ActivationArguments, ActivationKernel, ActivationType, Conv1dScanArguments,
    Conv1dScanKernel, DtDecayArguments, DtDecayKernel, SSDPrefillArguments,
    SSDPrefillKernel, SSDUpdateArguments, SSDUpdateKernel,
    SplitConvOutputsArguments, SplitConvOutputsKernel, SplitInProjArguments,
    SplitInProjKernel,
};
use crate::{
    DataType,
    backends::metal::{
        KernelDataType, MTLContext, MetalArray,
        compilation_parameters::CompilationConfig,
        forward_pass::{
            ArrayId, ForwardPassState,
            encodable_with_state::{EncodableWithState, EncodingParameters},
            transformer_layer,
        },
        kernel::TensorAddBias,
    },
    config::{Activation, DecoderLayerType, mamba::Mamba2Config},
    parameters::ParameterTree,
};

pub(crate) struct MambaMixerEncodable {
    layer_index: usize,
    config: Mamba2Config,
    activation: Activation,
    in_projection: Box<dyn EncodableWithState>,
    out_projection: Box<dyn EncodableWithState>,
    split_inproj: SplitInProjKernel,
    tensor_add_bias: TensorAddBias,
    dt_decay: DtDecayKernel,
    conv_scan: Conv1dScanKernel,
    activation_kernel: ActivationKernel,
    split_conv_outputs: SplitConvOutputsKernel,
    ssm_prefill: SSDPrefillKernel,
    ssd_update: SSDUpdateKernel,
    conv_weight: MetalArray,
    conv_bias: Option<MetalArray>,
    gate_bias: MetalArray,
    skip_connection_weight: MetalArray,
}

impl MambaMixerEncodable {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        mtl_context: &MTLContext,
        layer_type: DecoderLayerType,
        mamba_config: Mamba2Config,
        compilation_config: Rc<CompilationConfig>,
        layer_index: usize,
        model_dim: usize,
        num_heads: usize,
        head_dim: usize,
        num_groups: usize,
        decoder_layer_loader: &ParameterTree<Rc<MTLContext>>,
    ) -> Self {
        let _ = (num_heads, head_dim, num_groups);
        if !matches!(layer_type, DecoderLayerType::StateSpace { .. }) {
            panic!(
                "Layer {} marked as transformer but Mamba mixer config provided",
                layer_index
            );
        }
        let split_tree = resolve_subtree(decoder_layer_loader, &["mixer"]);
        let conv_tree = resolve_subtree(&split_tree, &["conv", "conv1d"]);

        let data_type: DataType =
            mamba_config.in_projection_config.activation_precision().into();
        let kernel_data_type: KernelDataType = data_type.into();
        let activation = mamba_config.conv_config.activation.clone();

        let in_projection = transformer_layer::linear_block(
            &mamba_config.in_projection_config,
            mamba_config.has_in_biases,
            model_dim,
            [
                mamba_config.conv_dim(),
                mamba_config.inner_dim(),
                mamba_config.num_value_heads,
            ],
            mtl_context,
            &resolve_subtree(
                decoder_layer_loader,
                &["mixer.in_projection", "mixer.in_proj"],
            ),
            ArrayId::Main,
            ArrayId::SsmInProj,
            &compilation_config.descriptor_mlp,
        );

        let out_projection = transformer_layer::linear_block(
            &mamba_config.out_projection_config,
            mamba_config.has_out_biases,
            mamba_config.inner_dim(),
            [model_dim],
            mtl_context,
            &resolve_subtree(
                decoder_layer_loader,
                &["mixer.out_projection", "mixer.out_proj"],
            ),
            ArrayId::AttentionOutput,
            ArrayId::Main,
            &compilation_config.descriptor_mlp,
        );

        let conv_weight = conv_tree.leaf("weight").unwrap().clone();
        let conv_bias = if mamba_config.conv_config.has_biases {
            Some(conv_tree.leaf("bias").unwrap().clone())
        } else {
            None
        };
        let gate_bias = split_tree.leaf("gate_bias").unwrap().clone();
        let skip_connection_weight =
            split_tree.leaf("skip_connection_weight").unwrap().clone();

        let split_inproj =
            SplitInProjKernel::new(mtl_context, kernel_data_type)
                .expect("Failed to create split in-projection kernel");
        let tensor_add_bias = TensorAddBias::new(mtl_context, kernel_data_type)
            .expect("Failed to create tensor add bias kernel");
        let dt_decay = DtDecayKernel::new(mtl_context, kernel_data_type)
            .expect("Failed to create dt/decay kernel");
        let conv_scan = Conv1dScanKernel::new(mtl_context, kernel_data_type)
            .expect("Failed to create conv scan kernel");
        let activation_kernel =
            ActivationKernel::new(mtl_context, kernel_data_type)
                .expect("Failed to create activation kernel");
        let split_conv_outputs =
            SplitConvOutputsKernel::new(mtl_context, kernel_data_type)
                .expect("Failed to create split conv outputs kernel");
        let ssm_prefill = SSDPrefillKernel::new(mtl_context, kernel_data_type)
            .expect("Failed to create SSD prefill kernel");
        let ssd_update = SSDUpdateKernel::new(mtl_context, kernel_data_type)
            .expect("Failed to create SSD decode kernel");

        Self {
            layer_index,
            config: mamba_config,
            activation,
            in_projection,
            out_projection,
            split_inproj,
            tensor_add_bias,
            dt_decay,
            conv_scan,
            activation_kernel,
            split_conv_outputs,
            ssm_prefill,
            ssd_update,
            conv_weight,
            conv_bias,
            gate_bias,
            skip_connection_weight,
        }
    }

    fn encode_pipeline(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        let suffix_length = state.aux_buffers_suffix_length();
        let active_suffix_length = state.active_suffix_length();
        if suffix_length == 0 || active_suffix_length == 0 {
            return;
        }

        self.in_projection.encode(state, command_buffer, parameters);
        self.split_inproj(state, command_buffer, suffix_length);
        self.add_gate_bias(state, command_buffer, suffix_length);
        self.run_dt_decay(state, command_buffer, suffix_length);
        self.run_conv_scan(state, command_buffer, suffix_length);
        self.apply_activation(state, command_buffer, suffix_length);
        self.split_conv_outputs(state, command_buffer, suffix_length);

        if suffix_length == 1 {
            self.run_decode_ssm(state, command_buffer, suffix_length);
        } else {
            self.run_prefill_ssm(state, command_buffer, active_suffix_length);
        }

        self.out_projection.encode(state, command_buffer, parameters);
    }

    fn split_inproj(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        suffix_length: usize,
    ) {
        let arrays = state.arrays(&[
            ArrayId::SsmInProj,
            ArrayId::SsmPacked(self.layer_index),
            ArrayId::SsmZ(self.layer_index),
            ArrayId::SsmDt(self.layer_index),
        ]);
        let mut in_proj = arrays[0].borrow_mut();
        let mut conv_inputs = arrays[1].borrow_mut();
        let mut gate = arrays[2].borrow_mut();
        let mut dt = arrays[3].borrow_mut();

        let input_buf = unsafe { in_proj.mtl_buffer().to_owned() };
        let conv_buf = unsafe { conv_inputs.mtl_buffer().to_owned() };
        let gate_buf = unsafe { gate.mtl_buffer().to_owned() };
        let dt_buf = unsafe { dt.mtl_buffer().to_owned() };

        let conv_dim = self.config.conv_dim();
        let inner_dim = self.config.inner_dim();
        let num_heads = self.config.num_value_heads;
        let total_dim = conv_dim + inner_dim + num_heads;

        let mtl_command_buffer =
            command_buffer.root_command_buffer().to_owned();
        let compute = mtl_command_buffer.new_compute_command_encoder();
        self.split_inproj
            .encode(
                &compute,
                SplitInProjArguments {
                    input: &input_buf,
                    conv_out: &conv_buf,
                    z_out: &gate_buf,
                    dt_out: &dt_buf,
                    total_dim,
                    conv_dim,
                    inner_dim,
                    num_heads,
                    suffix_length,
                },
            )
            .expect("Failed to encode split in-projection kernel");
        compute.end_encoding();
    }

    fn add_gate_bias(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        suffix_length: usize,
    ) {
        let gate_arrays = state.arrays(&[ArrayId::SsmZ(self.layer_index)]);
        let mut gate = gate_arrays[0].borrow_mut();
        let gate_buf = unsafe { gate.mtl_buffer().to_owned() };
        let mut gate_bias = self.gate_bias.clone();
        let bias_buf = unsafe { gate_bias.mtl_buffer().to_owned() };

        let inner_dim = self.config.inner_dim();
        let total = inner_dim * suffix_length;
        let cmd = command_buffer.root_command_buffer().to_owned();
        self.tensor_add_bias.encode_into_command_buffer(
            &gate_buf, &bias_buf, &gate_buf, inner_dim, total, &cmd,
        );
    }

    fn run_dt_decay(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        suffix_length: usize,
    ) {
        let arrays = state.arrays(&[
            ArrayId::SsmDt(self.layer_index),
            ArrayId::SsmDecay(self.layer_index),
        ]);
        let mut dt_arr = arrays[0].borrow_mut();
        let mut decay_arr = arrays[1].borrow_mut();
        let dt_buf = unsafe { dt_arr.mtl_buffer().to_owned() };
        let decay_buf = unsafe { decay_arr.mtl_buffer().to_owned() };
        let cmd = command_buffer.root_command_buffer().to_owned();
        let compute = cmd.new_compute_command_encoder();
        self.dt_decay
            .encode(
                &compute,
                DtDecayArguments {
                    dt: &dt_buf,
                    decay: &decay_buf,
                    num_heads: self.config.num_value_heads,
                    suffix_length,
                },
            )
            .expect("Failed to encode dt/decay kernel");
        compute.end_encoding();
    }

    fn run_conv_scan(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        suffix_length: usize,
    ) {
        let arrays = state.arrays(&[
            ArrayId::SsmPacked(self.layer_index),
            ArrayId::SsmConvState(self.layer_index),
        ]);
        let mut conv_inputs = arrays[0].borrow_mut();
        let mut conv_state = arrays[1].borrow_mut();
        let input_buf = unsafe { conv_inputs.mtl_buffer().to_owned() };
        let state_buf = unsafe { conv_state.mtl_buffer().to_owned() };
        let mut weight_storage = self.conv_weight.clone();
        let weight_buf = unsafe { weight_storage.mtl_buffer().to_owned() };
        let bias_buf = self.conv_bias.as_ref().map(|arr| {
            let mut storage = arr.clone();
            unsafe { storage.mtl_buffer().to_owned() }
        });

        let conv_dim = self.config.conv_dim();
        let state_stride =
            self.config.conv_config.kernel_size.saturating_sub(1);
        let cmd = command_buffer.root_command_buffer().to_owned();
        let compute = cmd.new_compute_command_encoder();
        self.conv_scan
            .encode(
                &compute,
                Conv1dScanArguments {
                    x: &input_buf,
                    w: &weight_buf,
                    b: bias_buf.as_ref(),
                    state: &state_buf,
                    y: &input_buf,
                    suffix_len: suffix_length,
                    kernel_size: self.config.conv_config.kernel_size as i32,
                    row_stride: conv_dim,
                    state_stride,
                    channels: conv_dim,
                },
            )
            .expect("Failed to encode conv scan kernel");
        compute.end_encoding();
    }

    fn apply_activation(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        suffix_length: usize,
    ) {
        let arrays = state.arrays(&[ArrayId::SsmPacked(self.layer_index)]);
        let mut conv_out = arrays[0].borrow_mut();
        let buf = unsafe { conv_out.mtl_buffer().to_owned() };
        let cmd = command_buffer.root_command_buffer().to_owned();
        let compute = cmd.new_compute_command_encoder();
        self.activation_kernel
            .encode(
                &compute,
                ActivationArguments {
                    data: &buf,
                    row_stride: self.config.conv_dim(),
                    suffix_length,
                    activation: self.activation_type(),
                },
            )
            .expect("Failed to encode activation kernel");
        compute.end_encoding();
    }

    fn split_conv_outputs(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        suffix_length: usize,
    ) {
        let arrays = state.arrays(&[
            ArrayId::SsmPacked(self.layer_index),
            ArrayId::SsmX(self.layer_index),
            ArrayId::SsmB(self.layer_index),
            ArrayId::SsmC(self.layer_index),
        ]);
        let mut conv = arrays[0].borrow_mut();
        let conv_buf = unsafe { conv.mtl_buffer().to_owned() };
        drop(conv);
        let mut x = arrays[1].borrow_mut();
        let x_buf = unsafe { x.mtl_buffer().to_owned() };
        drop(x);
        let mut b = arrays[2].borrow_mut();
        let b_buf = unsafe { b.mtl_buffer().to_owned() };
        drop(b);
        let mut c = arrays[3].borrow_mut();
        let c_buf = unsafe { c.mtl_buffer().to_owned() };
        drop(c);

        let conv_dim = self.config.conv_dim();
        let inner_dim = self.config.inner_dim();
        let proj_dim = self.config.num_groups * self.config.state_dim;
        let cmd = command_buffer.root_command_buffer().to_owned();
        let compute = cmd.new_compute_command_encoder();
        self.split_conv_outputs
            .encode(
                &compute,
                SplitConvOutputsArguments {
                    conv_input: &conv_buf,
                    x_out: &x_buf,
                    b_out: &b_buf,
                    c_out: &c_buf,
                    conv_dim,
                    inner_dim,
                    proj_dim,
                    suffix_length,
                },
            )
            .expect("Failed to encode split conv outputs kernel");
        compute.end_encoding();
    }

    fn run_prefill_ssm(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        suffix_length: usize,
    ) {
        let arrays = state.arrays(&[
            ArrayId::SsmX(self.layer_index),
            ArrayId::SsmB(self.layer_index),
            ArrayId::SsmC(self.layer_index),
            ArrayId::SsmDt(self.layer_index),
            ArrayId::SsmDecay(self.layer_index),
            ArrayId::SsmZ(self.layer_index),
            ArrayId::SsmState(self.layer_index),
            ArrayId::AttentionOutput,
        ]);
        let mut x = arrays[0].borrow_mut();
        let x_buf = unsafe { x.mtl_buffer().to_owned() };
        drop(x);
        let mut b = arrays[1].borrow_mut();
        let b_buf = unsafe { b.mtl_buffer().to_owned() };
        drop(b);
        let mut c = arrays[2].borrow_mut();
        let c_buf = unsafe { c.mtl_buffer().to_owned() };
        drop(c);
        let mut dt = arrays[3].borrow_mut();
        let dt_buf = unsafe { dt.mtl_buffer().to_owned() };
        drop(dt);
        let mut decay = arrays[4].borrow_mut();
        let decay_buf = unsafe { decay.mtl_buffer().to_owned() };
        drop(decay);
        let mut z = arrays[5].borrow_mut();
        let z_buf = unsafe { z.mtl_buffer().to_owned() };
        drop(z);
        let mut state_buf = arrays[6].borrow_mut();
        let state_raw = unsafe { state_buf.mtl_buffer().to_owned() };
        drop(state_buf);
        let mut out = arrays[7].borrow_mut();
        let out_buf = unsafe { out.mtl_buffer().to_owned() };
        let mut skip_weights = self.skip_connection_weight.clone();
        let skip = unsafe { skip_weights.mtl_buffer().to_owned() };

        let cmd = command_buffer.root_command_buffer().to_owned();
        let compute = cmd.new_compute_command_encoder();
        self.ssm_prefill
            .encode(
                &compute,
                SSDPrefillArguments {
                    x: &x_buf,
                    dt: &dt_buf,
                    decay: &decay_buf,
                    b: &b_buf,
                    c: &c_buf,
                    d: &skip,
                    z: &z_buf,
                    state: &state_raw,
                    y: &out_buf,
                    suffix_len: suffix_length,
                    group_size: (self.config.num_value_heads
                        / self.config.num_groups)
                        as i32,
                    state_size: self.config.state_dim as i32,
                    x_strides: [
                        self.config.num_value_heads * self.config.head_dim,
                        self.config.head_dim,
                        1,
                    ],
                    dt_strides: [self.config.num_value_heads, 1],
                    cb_strides: [
                        self.config.num_groups * self.config.state_dim,
                        self.config.state_dim,
                        1,
                    ],
                    state_strides: [
                        self.config.head_dim * self.config.state_dim,
                        self.config.state_dim,
                        1,
                    ],
                    channels: self.config.num_value_heads,
                    head_dim: self.config.head_dim,
                },
            )
            .expect("Failed to encode SSD prefill kernel");
        compute.end_encoding();
    }

    fn run_decode_ssm(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        suffix_length: usize,
    ) {
        let arrays = state.arrays(&[
            ArrayId::SsmX(self.layer_index),
            ArrayId::SsmB(self.layer_index),
            ArrayId::SsmC(self.layer_index),
            ArrayId::SsmDt(self.layer_index),
            ArrayId::SsmDecay(self.layer_index),
            ArrayId::SsmZ(self.layer_index),
            ArrayId::SsmState(self.layer_index),
            ArrayId::AttentionOutput,
        ]);
        let mut x = arrays[0].borrow_mut();
        let x_buf = unsafe { x.mtl_buffer().to_owned() };
        drop(x);
        let mut b = arrays[1].borrow_mut();
        let b_buf = unsafe { b.mtl_buffer().to_owned() };
        drop(b);
        let mut c = arrays[2].borrow_mut();
        let c_buf = unsafe { c.mtl_buffer().to_owned() };
        drop(c);
        let mut dt = arrays[3].borrow_mut();
        let dt_buf = unsafe { dt.mtl_buffer().to_owned() };
        drop(dt);
        let mut decay = arrays[4].borrow_mut();
        let decay_buf = unsafe { decay.mtl_buffer().to_owned() };
        drop(decay);
        let mut z = arrays[5].borrow_mut();
        let z_buf = unsafe { z.mtl_buffer().to_owned() };
        drop(z);
        let mut state_arr = arrays[6].borrow_mut();
        let state_buf = unsafe { state_arr.mtl_buffer().to_owned() };
        drop(state_arr);
        let mut y = arrays[7].borrow_mut();
        let y_buf = unsafe { y.mtl_buffer().to_owned() };
        let mut skip_weights = self.skip_connection_weight.clone();
        let skip_buf = unsafe { skip_weights.mtl_buffer().to_owned() };

        let h = self.config.num_value_heads;
        let g = self.config.num_groups;
        let dh = self.config.head_dim;
        let n = self.config.state_dim;

        let x_strides = [h * dh, dh, 1usize];
        let dt_strides = [h, 1usize];
        let cb_strides = [g * n, n, 1usize];
        let state_strides = [h * dh * n, dh * n, n, 1usize];
        let group_size = (h / g) as i32;
        let state_size = n as i32;

        let cmd = command_buffer.root_command_buffer().to_owned();
        let compute = cmd.new_compute_command_encoder();
        self.ssd_update
            .encode(
                &compute,
                SSDUpdateArguments {
                    x: &x_buf,
                    dt: &dt_buf,
                    decay: &decay_buf,
                    b: &b_buf,
                    c: &c_buf,
                    d: &skip_buf,
                    z: &z_buf,
                    state: &state_buf,
                    y: &y_buf,
                    next_state: &state_buf,
                    group_size,
                    state_size,
                    x_strides,
                    dt_strides,
                    cb_strides,
                    state_strides,
                    b_size: suffix_length,
                    h_size: h,
                    dh_size: dh,
                },
            )
            .expect("Failed to encode SSD decode kernel");
        compute.end_encoding();
    }

    fn activation_type(&self) -> ActivationType {
        match self.activation {
            Activation::SILU {
                ..
            } => ActivationType::Silu,
            Activation::GELU => ActivationType::Gelu,
            Activation::Identity => ActivationType::Identity,
        }
    }
}

impl EncodableWithState for MambaMixerEncodable {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        self.encode_pipeline(state, command_buffer, parameters);
    }
}

fn resolve_subtree<'tree>(
    loader: &'tree ParameterTree<Rc<MTLContext>>,
    candidates: &[&str],
) -> ParameterTree<'tree, Rc<MTLContext>> {
    for candidate in candidates {
        if let Ok(tree) = loader.subtree(candidate) {
            return tree;
        }
    }

    let missing =
        candidates.first().copied().unwrap_or("<missing subtree name>");
    loader.subtree(missing).unwrap_or_else(|_| {
        panic!("Unable to resolve parameter subtree '{missing}'")
    })
}
