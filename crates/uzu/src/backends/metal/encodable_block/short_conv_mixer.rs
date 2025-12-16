use std::rc::Rc;

use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{EncodableBlock, EncodingParameters, transformer_layer};
use crate::{
    DataType,
    backends::metal::{
        KernelDataType, MTLContext, MetalArray,
        compilation_parameters::CompilationConfig,
        forward_pass::{ArrayId, ForwardPassState},
        kernel::short_conv::{
            ShortConvDecodeArguments, ShortConvKernel,
            ShortConvPrefillArguments,
        },
    },
    config::{DecoderLayerType, ShortConvConfig},
    parameters::ParameterTree,
};

pub(crate) struct ShortConvMixer {
    layer_index: usize,
    config: ShortConvConfig,
    model_dim: usize,
    in_projection: Box<dyn EncodableBlock>,
    out_projection: Box<dyn EncodableBlock>,
    short_conv_kernel: ShortConvKernel,
    conv_weight: MetalArray,
    conv_bias: Option<MetalArray>,
}

fn resolve_subtree<'tree>(
    tree: &'tree ParameterTree<Rc<MTLContext>>,
    candidates: &[&str],
) -> ParameterTree<'tree, Rc<MTLContext>> {
    for candidate in candidates {
        if let Ok(subtree) = tree.subtree(candidate) {
            return subtree;
        }
    }
    panic!("Could not find any of {:?} in parameter tree", candidates);
}

impl ShortConvMixer {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        mtl_context: &MTLContext,
        layer_type: DecoderLayerType,
        short_conv_config: ShortConvConfig,
        compilation_config: Rc<CompilationConfig>,
        layer_index: usize,
        model_dim: usize,
        decoder_layer_loader: &ParameterTree<Rc<MTLContext>>,
    ) -> Self {
        if !matches!(layer_type, DecoderLayerType::ShortConv { .. }) {
            panic!(
                "Layer {} marked as non-ShortConv but ShortConv config provided",
                layer_index
            );
        }

        let mixer_tree = resolve_subtree(decoder_layer_loader, &["mixer"]);
        let conv_tree = resolve_subtree(&mixer_tree, &["conv"]);

        let data_type: DataType = short_conv_config
            .in_projection_config
            .activation_precision()
            .into();
        let kernel_data_type: KernelDataType = data_type.into();

        let in_projection = transformer_layer::linear_block(
            &short_conv_config.in_projection_config,
            false,
            model_dim,
            [model_dim * 3],
            mtl_context,
            &resolve_subtree(&mixer_tree, &["in_projection", "in_proj"]),
            ArrayId::Main,
            ArrayId::SsmInProj,
            &compilation_config.descriptor_mlp,
        );

        let out_projection = transformer_layer::linear_block(
            &short_conv_config.out_projection_config,
            false,
            model_dim,
            [model_dim],
            mtl_context,
            &resolve_subtree(&mixer_tree, &["out_projection", "out_proj"]),
            ArrayId::AttentionOutput,
            ArrayId::Main,
            &compilation_config.descriptor_mlp,
        );

        let conv_weight = conv_tree.leaf("weights").unwrap().clone();
        let conv_bias = if short_conv_config.conv_config.has_biases {
            Some(conv_tree.leaf("biases").unwrap().clone())
        } else {
            None
        };

        let short_conv_kernel =
            ShortConvKernel::new(mtl_context, kernel_data_type)
                .expect("Failed to create short conv kernel");

        Self {
            layer_index,
            config: short_conv_config,
            model_dim,
            in_projection,
            out_projection,
            short_conv_kernel,
            conv_weight,
            conv_bias,
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

        if suffix_length == 1 {
            self.run_decode_conv(state, command_buffer, active_suffix_length);
        } else {
            self.run_prefill_conv(state, command_buffer, active_suffix_length);
        }

        self.out_projection.encode(state, command_buffer, parameters);

        if parameters.wait_until_completed {
            let mtl_command_buffer =
                command_buffer.root_command_buffer().to_owned();
            command_buffer.commit_and_continue();
            mtl_command_buffer.wait_until_completed();
        }
    }

    fn run_prefill_conv(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        suffix_length: usize,
    ) {
        let arrays = state.arrays(&[
            ArrayId::SsmInProj,
            ArrayId::ShortConvState(self.layer_index),
            ArrayId::AttentionOutput,
        ]);
        let mut in_proj = arrays[0].borrow_mut();
        let mut conv_state = arrays[1].borrow_mut();
        let mut out = arrays[2].borrow_mut();

        let in_proj_buf = unsafe { in_proj.mtl_buffer().to_owned() };
        let state_buf = unsafe { conv_state.mtl_buffer().to_owned() };
        let out_buf = unsafe { out.mtl_buffer().to_owned() };

        let mut conv_weight = self.conv_weight.clone();
        let weight_buf = unsafe { conv_weight.mtl_buffer().to_owned() };
        let bias_buf = self.conv_bias.as_ref().map(|b| {
            let mut b = b.clone();
            unsafe { b.mtl_buffer().to_owned() }
        });

        let kernel_size = self.config.kernel_size;
        let state_stride = kernel_size.saturating_sub(1);

        let mtl_command_buffer =
            command_buffer.root_command_buffer().to_owned();
        let compute = mtl_command_buffer.new_compute_command_encoder();

        self.short_conv_kernel
            .encode_prefill(
                &compute,
                ShortConvPrefillArguments {
                    in_proj: &in_proj_buf,
                    w: &weight_buf,
                    b: bias_buf.as_ref(),
                    state_in: &state_buf,
                    out: &out_buf,
                    state_out: &state_buf,
                    suffix_len: suffix_length,
                    kernel_size: kernel_size as i32,
                    in_proj_stride: self.model_dim * 3,
                    state_stride,
                    model_dim: self.model_dim,
                },
            )
            .expect("Failed to encode short conv prefill kernel");

        compute.end_encoding();
    }

    fn run_decode_conv(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        suffix_length: usize,
    ) {
        let arrays = state.arrays(&[
            ArrayId::SsmInProj,
            ArrayId::ShortConvState(self.layer_index),
            ArrayId::AttentionOutput,
        ]);
        let mut in_proj = arrays[0].borrow_mut();
        let mut conv_state = arrays[1].borrow_mut();
        let mut out = arrays[2].borrow_mut();

        let in_proj_buf = unsafe { in_proj.mtl_buffer().to_owned() };
        let state_buf = unsafe { conv_state.mtl_buffer().to_owned() };
        let out_buf = unsafe { out.mtl_buffer().to_owned() };

        let mut conv_weight = self.conv_weight.clone();
        let weight_buf = unsafe { conv_weight.mtl_buffer().to_owned() };
        let bias_buf = self.conv_bias.as_ref().map(|b| {
            let mut b = b.clone();
            unsafe { b.mtl_buffer().to_owned() }
        });

        let kernel_size = self.config.kernel_size;
        let state_stride = kernel_size.saturating_sub(1);

        let mtl_command_buffer =
            command_buffer.root_command_buffer().to_owned();
        let compute = mtl_command_buffer.new_compute_command_encoder();

        self.short_conv_kernel
            .encode_decode(
                &compute,
                ShortConvDecodeArguments {
                    in_proj: &in_proj_buf,
                    w: &weight_buf,
                    b: bias_buf.as_ref(),
                    state: &state_buf,
                    out: &out_buf,
                    next_state: &state_buf,
                    suffix_len: suffix_length,
                    kernel_size: kernel_size as i32,
                    in_proj_stride: self.model_dim * 3,
                    state_stride,
                    model_dim: self.model_dim,
                },
            )
            .expect("Failed to encode short conv decode kernel");

        compute.end_encoding();
    }
}

impl EncodableBlock for ShortConvMixer {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        self.encode_pipeline(state, command_buffer, parameters);
    }
}
