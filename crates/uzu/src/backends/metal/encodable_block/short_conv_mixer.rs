use std::rc::Rc;

use super::{EncodableBlock, Metal, transformer_layer};
use crate::{
    DataType,
    backends::{
        common::Context,
        metal::{
            MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLContext, MetalArray, ProtocolObject,
            Retained,
            compilation_parameters::CompilationConfig,
            kernel::short_conv::{
                ShortConvDecodeArguments, ShortConvKernel, ShortConvPackArguments, ShortConvPrefillArguments,
                ShortConvTrieArguments,
            },
        },
    },
    config::{DecoderLayerType, ShortConvConfig},
    encodable_block::EncodingParameters,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::ParameterTree,
};

pub(crate) struct ShortConvMixer {
    layer_index: usize,
    config: ShortConvConfig,
    model_dim: usize,
    in_projection: Box<dyn EncodableBlock<Metal>>,
    out_projection: Box<dyn EncodableBlock<Metal>>,
    short_conv_kernel: ShortConvKernel,
    conv_weight: MetalArray,
    conv_bias: Option<MetalArray>,
}

fn resolve_subtree<'tree>(
    tree: &'tree ParameterTree<MTLContext>,
    candidates: &[&str],
) -> ParameterTree<'tree, MTLContext> {
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
        decoder_layer_loader: &ParameterTree<MTLContext>,
    ) -> Self {
        if !matches!(layer_type, DecoderLayerType::ShortConv { .. }) {
            panic!("Layer {} marked as non-ShortConv but ShortConv config provided", layer_index);
        }

        let mixer_tree = resolve_subtree(decoder_layer_loader, &["mixer"]);
        let conv_tree = resolve_subtree(&mixer_tree, &["conv"]);

        let data_type: DataType = short_conv_config.in_projection_config.activation_precision().into();

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

        let short_conv_kernel =
            ShortConvKernel::new(mtl_context, data_type).expect("Failed to create short conv kernel");

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
        state: &mut ForwardPassState<Metal>,
        parameters: &EncodingParameters<Metal>,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    ) {
        let active_suffix_length = state.active_suffix_length();
        if active_suffix_length == 0 {
            return;
        }

        self.in_projection.encode(state, parameters, command_buffer);

        let encoder = command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
        self.run_conv(state, &encoder, active_suffix_length);
        encoder.end_encoding();

        self.out_projection.encode(state, parameters, command_buffer);

        if parameters.wait_until_completed {
            command_buffer.commit();
            command_buffer.wait_until_completed();
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

        self.run_conv(state, encoder, active_suffix_length);

        self.out_projection.encode_with_shared_encoder(state, parameters, encoder);
    }

    fn clear_suffix_state_valid_range(
        &self,
        state: &ForwardPassState<Metal>,
    ) {
        let Some(cache_layers) = state.cache_layers() else {
            return;
        };
        let cache = cache_layers.borrow();
        let layer = cache.data[self.layer_index].as_short_conv().expect("Expected ShortConv layer");
        layer.clear_suffix_state_valid_range();
    }

    fn set_suffix_state_valid_range(
        &self,
        state: &ForwardPassState<Metal>,
        start: usize,
        len: usize,
    ) {
        let Some(cache_layers) = state.cache_layers() else {
            return;
        };
        let cache = cache_layers.borrow();
        let layer = cache.data[self.layer_index].as_short_conv().expect("Expected ShortConv layer");
        layer.set_suffix_state_valid_range(start, len);
    }

    fn run_conv(
        &self,
        state: &mut ForwardPassState<Metal>,
        compute: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        active_suffix_length: usize,
    ) {
        self.clear_suffix_state_valid_range(state);

        if active_suffix_length == 1 {
            self.run_decode_conv(state, compute, 1);
            return;
        }

        let sampling_len = state.sampling_length();
        if sampling_len == 0 {
            self.run_prefill_conv(state, compute, active_suffix_length);
            return;
        }

        let sampling_start = state.sampling_start();
        let trie_len = sampling_len;

        if trie_len <= 1 {
            self.run_prefill_conv(state, compute, active_suffix_length);
            return;
        }

        if sampling_start > 0 {
            if sampling_start == 1 {
                self.run_decode_conv(state, compute, 1);
            } else {
                self.run_prefill_conv(state, compute, sampling_start);
            }
        }

        self.run_trie_conv(state, compute, sampling_start, trie_len);
        self.set_suffix_state_valid_range(state, sampling_start, trie_len);
    }

    fn run_prefill_conv(
        &self,
        state: &mut ForwardPassState<Metal>,
        compute: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        suffix_length: usize,
    ) {
        let arrays =
            state.arrays(&[ArrayId::SsmInProj, ArrayId::ShortConvState(self.layer_index), ArrayId::AttentionOutput]);
        let in_proj = arrays[0].borrow_mut();
        let conv_state = arrays[1].borrow_mut();
        let out = arrays[2].borrow_mut();

        let in_proj_buf = in_proj.buffer().clone();
        let state_buf = conv_state.buffer().clone();
        let out_buf = out.buffer().clone();

        let conv_weight = self.conv_weight.clone();
        let weight_buf = conv_weight.buffer().clone();
        let bias_buf = self.conv_bias.as_ref().map(|b| {
            let b = b.clone();
            b.buffer().clone()
        });

        let kernel_size = self.config.kernel_size;
        let state_stride = kernel_size.saturating_sub(1);

        // Allocate temporary padded buffer
        let data_type: DataType = self.config.in_projection_config.activation_precision().into();
        let element_size = data_type.size_in_bytes();
        let padded_rows = state_stride + suffix_length;
        let padded_size = padded_rows * self.model_dim * element_size;
        let padded_buf = state.mtl_context().create_buffer(padded_size).expect("Failed to create padded buffer");
        self.short_conv_kernel
            .encode_pack(
                compute,
                ShortConvPackArguments {
                    state_in: &state_buf,
                    in_proj: &in_proj_buf,
                    padded: &padded_buf,
                    state_stride,
                    suffix_len: suffix_length,
                    in_proj_stride: self.model_dim * 3,
                    model_dim: self.model_dim,
                },
            )
            .expect("Failed to encode short conv pack kernel");

        self.short_conv_kernel
            .encode_prefill(
                compute,
                ShortConvPrefillArguments {
                    padded: &padded_buf,
                    in_proj: &in_proj_buf,
                    w: &weight_buf,
                    b: bias_buf.as_deref(),
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
    }

    fn run_trie_conv(
        &self,
        state: &mut ForwardPassState<Metal>,
        compute: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        sampling_start: usize,
        trie_len: usize,
    ) {
        if trie_len == 0 {
            return;
        }

        let arrays = state.arrays(&[
            ArrayId::SsmInProj,
            ArrayId::TokenParents,
            ArrayId::ShortConvState(self.layer_index),
            ArrayId::ShortConvSuffixState(self.layer_index),
            ArrayId::AttentionOutput,
        ]);

        let in_proj = arrays[0].borrow_mut();
        let parents = arrays[1].borrow_mut();
        let conv_state = arrays[2].borrow_mut();
        let suffix_state = arrays[3].borrow_mut();
        let out = arrays[4].borrow_mut();

        let in_proj_buf = in_proj.buffer().clone();
        let parents_buf = parents.buffer().clone();
        let base_state_buf = conv_state.buffer().clone();
        let suffix_state_buf = suffix_state.buffer().clone();
        let out_buf = out.buffer().clone();

        let data_type: DataType = self.config.in_projection_config.activation_precision().into();
        let elem_bytes = data_type.size_in_bytes();

        let kernel_size = self.config.kernel_size;
        let state_stride = kernel_size.saturating_sub(1);
        let in_proj_stride = self.model_dim * 3;

        let in_proj_offset = in_proj.offset() + sampling_start * in_proj_stride * elem_bytes;
        let out_offset = out.offset() + sampling_start * self.model_dim * elem_bytes;
        let suffix_state_offset = suffix_state.offset() + sampling_start * self.model_dim * state_stride * elem_bytes;
        let base_state_offset = conv_state.offset();
        let parents_offset = parents.offset() + sampling_start * std::mem::size_of::<i32>();

        let conv_weight = self.conv_weight.clone();
        let weight_buf = conv_weight.buffer().clone();
        let bias_buf = self.conv_bias.as_ref().map(|b| {
            let b = b.clone();
            b.buffer().clone()
        });

        self.short_conv_kernel
            .encode_trie(
                compute,
                ShortConvTrieArguments {
                    in_proj: &in_proj_buf,
                    in_proj_offset,
                    w: &weight_buf,
                    b: bias_buf.as_deref(),
                    base_state: &base_state_buf,
                    base_state_offset,
                    parents: &parents_buf,
                    parents_offset,
                    out: &out_buf,
                    out_offset,
                    suffix_state: &suffix_state_buf,
                    suffix_state_offset,
                    suffix_len: trie_len,
                    kernel_size: kernel_size as i32,
                    in_proj_stride,
                    state_stride,
                    model_dim: self.model_dim,
                },
            )
            .expect("Failed to encode short conv trie kernel");
    }

    fn run_decode_conv(
        &self,
        state: &mut ForwardPassState<Metal>,
        compute: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        suffix_length: usize,
    ) {
        let arrays =
            state.arrays(&[ArrayId::SsmInProj, ArrayId::ShortConvState(self.layer_index), ArrayId::AttentionOutput]);
        let in_proj = arrays[0].borrow_mut();
        let conv_state = arrays[1].borrow_mut();
        let out = arrays[2].borrow_mut();

        let in_proj_buf = in_proj.buffer().clone();
        let state_buf = conv_state.buffer().clone();
        let out_buf = out.buffer().clone();

        let conv_weight = self.conv_weight.clone();
        let weight_buf = conv_weight.buffer().clone();
        let bias_buf = self.conv_bias.as_ref().map(|b| {
            let b = b.clone();
            b.buffer().clone()
        });

        let kernel_size = self.config.kernel_size;
        let state_stride = kernel_size.saturating_sub(1);

        self.short_conv_kernel
            .encode_decode(
                compute,
                ShortConvDecodeArguments {
                    in_proj: &in_proj_buf,
                    w: &weight_buf,
                    b: bias_buf.as_deref(),
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
    }
}

impl EncodableBlock<Metal> for ShortConvMixer {
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
        } else {
            self.encode_pipeline(state, parameters, command_buffer);
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
