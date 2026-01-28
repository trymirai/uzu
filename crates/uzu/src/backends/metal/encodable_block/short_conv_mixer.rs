use std::rc::Rc;

use super::{EncodableBlock, EncodingParameters, transformer_layer};
use crate::{
    DataType,
    DeviceContext,
    backends::metal::{
        KernelDataType, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLContext,
        MTLDeviceExt, MTLResource, MTLResourceOptions, MetalArray, ProtocolObject,
        compilation_parameters::CompilationConfig,
        forward_pass::{ArrayId, ForwardPassState},
        kernel::dsl::{ShortConvDecodeKernel, ShortConvPackKernel, ShortConvPrefillKernel}
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
    short_conv_pack_kernel: ShortConvPackKernel,
    short_conv_prefill_kernel: ShortConvPrefillKernel,
    short_conv_decode_kernel: ShortConvDecodeKernel,
    conv_weight: MetalArray,
    conv_bias: Option<MetalArray>,
    empty_array: MetalArray,
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
        _compilation_config: Rc<CompilationConfig>,
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
        )
        .expect("Failed to create in-projection kernel");

        let out_projection = transformer_layer::linear_block(
            &short_conv_config.out_projection_config,
            false,
            model_dim,
            [model_dim],
            mtl_context,
            &resolve_subtree(&mixer_tree, &["out_projection", "out_proj"]),
            ArrayId::AttentionOutput,
            ArrayId::Main,
        )
        .expect("Failed to create out-projection kernel");

        let conv_weight = conv_tree.leaf("weights").unwrap().clone();
        let conv_bias = if short_conv_config.conv_config.has_biases {
            Some(conv_tree.leaf("biases").unwrap().clone())
        } else {
            None
        };

        let has_bias = short_conv_config.conv_config.has_biases;
        let short_conv_pack_kernel =
            ShortConvPackKernel::new(mtl_context, kernel_data_type)
                .expect("Failed to create short_conv_pack_kernel");
        let short_conv_prefill_kernel =
            ShortConvPrefillKernel::new(mtl_context, kernel_data_type, has_bias)
                .expect("Failed to create short_conv_prefill_kernel");
        let short_conv_decode_kernel =
            ShortConvDecodeKernel::new(mtl_context, kernel_data_type, has_bias)
                .expect("Failed to create short_conv_decode_kernel");
        let empty_array = unsafe {
            mtl_context.array_uninitialized(&[0, 0, 0], data_type, "Empty".to_string())
        };
        Self {
            layer_index,
            config: short_conv_config,
            model_dim,
            in_projection,
            out_projection,
            short_conv_pack_kernel,
            short_conv_prefill_kernel,
            short_conv_decode_kernel,
            conv_weight,
            conv_bias,
            empty_array
        }
    }

    fn encode_pipeline(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        parameters: &EncodingParameters,
    ) {
        let active_suffix_length = state.active_suffix_length();
        if active_suffix_length == 0 {
            return;
        }

        self.in_projection.encode(state, command_buffer, parameters);

        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        if active_suffix_length == 1 {
            self.run_decode_conv(state, &encoder, active_suffix_length);
        } else {
            self.run_prefill_conv(state, &encoder, active_suffix_length);
        }
        encoder.end_encoding();

        self.out_projection.encode(state, command_buffer, parameters);

        if parameters.wait_until_completed {
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
    }

    fn encode_pipeline_with_encoder(
        &self,
        state: &mut ForwardPassState,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        parameters: &EncodingParameters,
    ) {
        let active_suffix_length = state.active_suffix_length();
        if active_suffix_length == 0 {
            return;
        }

        self.in_projection
            .encode_with_shared_encoder(state, encoder, parameters);

        if active_suffix_length == 1 {
            self.run_decode_conv(state, encoder, active_suffix_length);
        } else {
            self.run_prefill_conv(state, encoder, active_suffix_length);
        }

        self.out_projection
            .encode_with_shared_encoder(state, encoder, parameters);
    }

    fn run_prefill_conv(
        &self,
        state: &mut ForwardPassState,
        compute: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        suffix_length: usize,
    ) {
        let arrays = state.arrays(&[
            ArrayId::SsmInProj,
            ArrayId::ShortConvState(self.layer_index),
            ArrayId::AttentionOutput,
        ]);
        let in_proj = arrays[0].borrow_mut();
        let conv_state = arrays[1].borrow_mut();
        let out = arrays[2].borrow_mut();

        let in_proj_buf = in_proj.mtl_buffer_cloned();
        let state_buf = conv_state.mtl_buffer_cloned();
        let out_buf = out.mtl_buffer_cloned();

        let conv_weight = self.conv_weight.clone();
        let weight_buf = conv_weight.mtl_buffer_cloned();
        let bias_buf = match &self.conv_bias {
            Some(buf) => buf.clone().mtl_buffer_cloned(),
            None => self.empty_array.clone().mtl_buffer_cloned(),
        };

        let kernel_size = self.config.kernel_size;
        let state_stride = kernel_size.saturating_sub(1);

        // Allocate temporary padded buffer
        let data_type: DataType =
            self.config.in_projection_config.activation_precision().into();
        let element_size = data_type.size_in_bytes();
        let padded_rows = state_stride + suffix_length;
        let padded_size = padded_rows * self.model_dim * element_size;
        let device = in_proj_buf.device();
        let padded_buf = device
            .new_buffer(padded_size, MTLResourceOptions::STORAGE_MODE_PRIVATE)
            .expect("Failed to create padded buffer");

        self.short_conv_pack_kernel.encode(
            &state_buf,
            &in_proj_buf,
            &padded_buf,
            state_stride as u32,
            suffix_length as u32,
            (self.model_dim * 3) as u32,
            self.model_dim as u32,
            &compute
        );

        self.short_conv_prefill_kernel.encode(
            &padded_buf,
            &in_proj_buf,
            &weight_buf,
            &bias_buf,
            &out_buf,
            &state_buf,
            suffix_length as u32,
            kernel_size as u32,
            (self.model_dim * 3) as u32,
            state_stride as u32,
            self.model_dim as u32,
            &compute
        );
    }

    fn run_decode_conv(
        &self,
        state: &mut ForwardPassState,
        compute: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        suffix_length: usize,
    ) {
        let arrays = state.arrays(&[
            ArrayId::SsmInProj,
            ArrayId::ShortConvState(self.layer_index),
            ArrayId::AttentionOutput,
        ]);
        let in_proj = arrays[0].borrow_mut();
        let conv_state = arrays[1].borrow_mut();
        let out = arrays[2].borrow_mut();

        let in_proj_buf = in_proj.mtl_buffer_cloned();
        let state_buf = conv_state.mtl_buffer_cloned();
        let out_buf = out.mtl_buffer_cloned();

        let conv_weight = self.conv_weight.clone();
        let weight_buf = conv_weight.mtl_buffer_cloned();
        let bias_buf = match &self.conv_bias {
            Some(buf) => buf.clone().mtl_buffer_cloned(),
            None => self.empty_array.clone().mtl_buffer_cloned(),
        };

        let kernel_size = self.config.kernel_size;
        let state_stride = kernel_size.saturating_sub(1);

        self.short_conv_decode_kernel.encode(
            &in_proj_buf,
            &weight_buf,
            &bias_buf,
            &state_buf,
            &out_buf,
            &state_buf,
            suffix_length as u32,
            kernel_size as u32,
            (self.model_dim * 3) as u32,
            state_stride as u32,
            self.model_dim as u32,
            compute
        );
    }
}

impl EncodableBlock for ShortConvMixer {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        parameters: &EncodingParameters,
    ) {
        if self.supports_shared_encoder() {
            let encoder = command_buffer
                .new_compute_command_encoder()
                .expect("Failed to create compute command encoder");
            self.encode_pipeline_with_encoder(state, &encoder, parameters);
            encoder.end_encoding();

            if parameters.wait_until_completed {
                command_buffer.commit();
                command_buffer.wait_until_completed();
            }
        } else {
            self.encode_pipeline(state, command_buffer, parameters);
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        self.in_projection.supports_shared_encoder()
            && self.out_projection.supports_shared_encoder()
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        parameters: &EncodingParameters,
    ) {
        self.encode_pipeline_with_encoder(state, encoder, parameters);
    }
}
