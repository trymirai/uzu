use std::{
    cell::RefCell,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use crate::{
    DataType,
    array::Array,
    backends::common::{
        Backend, Context, Encoder, Kernels,
        kernel::{ShortConvDecodeKernel, ShortConvPackKernel, ShortConvPrefillKernel, ShortConvTrieKernel},
    },
    config::{DecoderLayerType, ShortConvConfig},
    encodable_block::linear::Linear,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterTree, resolve_subtree},
};

pub struct ShortConvMixer<B: Backend> {
    layer_index: usize,
    config: ShortConvConfig,
    model_dim: usize,
    in_projection: Box<dyn Linear<B>>,
    out_projection: Box<dyn Linear<B>>,
    short_conv_pack: <B::Kernels as Kernels>::ShortConvPackKernel,
    short_conv_prefill: <B::Kernels as Kernels>::ShortConvPrefillKernel,
    short_conv_decode: <B::Kernels as Kernels>::ShortConvDecodeKernel,
    short_conv_trie: <B::Kernels as Kernels>::ShortConvTrieKernel,
    conv_weight: Array<B>,
    conv_bias: Option<Array<B>>,
}

impl<B: Backend> ShortConvMixer<B> {
    pub fn new(
        context: &B::Context,
        layer_type: DecoderLayerType,
        short_conv_config: ShortConvConfig,
        layer_index: usize,
        model_dim: usize,
        decoder_layer_loader: &ParameterTree<B::Context>,
    ) -> (Self, Option<Rc<RefCell<B::Buffer>>>) {
        if !matches!(layer_type, DecoderLayerType::ShortConv { .. }) {
            panic!("Layer {} marked as non-ShortConv but ShortConv config provided", layer_index);
        }

        let mixer_tree = resolve_subtree(decoder_layer_loader, &["mixer"]);
        let conv_tree = resolve_subtree(&mixer_tree, &["conv"]);

        let data_type: DataType = short_conv_config.in_projection_config.activation_precision().into();

        let (in_projection, in_proj_input_hadamard_factors) = <dyn Linear<B>>::new_extracting_input_hadamard(
            &short_conv_config.in_projection_config,
            false,
            model_dim,
            [model_dim * 3],
            context,
            &resolve_subtree(&mixer_tree, &["in_projection", "in_proj"]),
            ArrayId::Main,
            ArrayId::SsmInProj,
        )
        .expect("Failed to create in-projection kernel");

        let out_projection = <dyn Linear<B>>::new(
            &short_conv_config.out_projection_config,
            false,
            model_dim,
            [model_dim],
            context,
            &resolve_subtree(&mixer_tree, &["out_projection", "out_proj"]),
            ArrayId::AttentionOutput,
            ArrayId::Main,
        )
        .expect("Failed to create out-projection kernel");

        let conv_weight = conv_tree.leaf_array("weights").unwrap().clone();
        let conv_bias = if short_conv_config.conv_config.has_biases {
            Some(conv_tree.leaf_array("biases").unwrap().clone())
        } else {
            None
        };

        let has_bias = short_conv_config.conv_config.has_biases;
        let short_conv_pack = <B::Kernels as Kernels>::ShortConvPackKernel::new(context, data_type)
            .expect("Failed to create short conv pack kernel");
        let short_conv_prefill = <B::Kernels as Kernels>::ShortConvPrefillKernel::new(context, data_type, has_bias)
            .expect("Failed to create short conv prefill kernel");
        let short_conv_decode = <B::Kernels as Kernels>::ShortConvDecodeKernel::new(context, data_type, has_bias, true)
            .expect("Failed to create short conv decode kernel");
        let short_conv_trie = <B::Kernels as Kernels>::ShortConvTrieKernel::new(context, data_type, has_bias)
            .expect("Failed to create short conv trie kernel");

        (
            Self {
                layer_index,
                config: short_conv_config,
                model_dim,
                in_projection,
                out_projection,
                short_conv_pack,
                short_conv_prefill,
                short_conv_decode,
                short_conv_trie,
                conv_weight,
                conv_bias,
            },
            in_proj_input_hadamard_factors,
        )
    }

    fn clear_suffix_state_valid_range(
        &self,
        state: &ForwardPassState<B>,
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
        state: &ForwardPassState<B>,
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
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
        active_suffix_length: usize,
    ) {
        self.clear_suffix_state_valid_range(state);

        if active_suffix_length == 1 {
            self.run_decode_conv(state, encoder, 1);
            return;
        }

        let sampling_len = state.sampling_length();
        if sampling_len == 0 {
            self.run_prefill_conv(state, encoder, active_suffix_length);
            return;
        }

        let sampling_start = state.sampling_start();
        let trie_len = sampling_len;

        if trie_len <= 1 {
            self.run_prefill_conv(state, encoder, active_suffix_length);
            return;
        }

        if sampling_start > 0 {
            if sampling_start == 1 {
                self.run_decode_conv(state, encoder, 1);
            } else {
                self.run_prefill_conv(state, encoder, sampling_start);
            }
        }

        self.run_trie_conv(state, encoder, sampling_start, trie_len);
        self.set_suffix_state_valid_range(state, sampling_start, trie_len);
    }

    fn run_prefill_conv(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
        suffix_length: usize,
    ) {
        if self.model_dim == 0 || suffix_length == 0 {
            return;
        }

        let in_proj = state.array(ArrayId::SsmInProj);
        let conv_state = state.array(ArrayId::ShortConvState(self.layer_index));
        let out = state.array(ArrayId::AttentionOutput);

        let bias_buf_rc = self.conv_bias.as_ref().map(|b| b.buffer());
        let bias_buf_borrow = bias_buf_rc.as_ref().map(|rc| rc.borrow());

        let kernel_size = self.config.kernel_size;
        let state_stride = kernel_size.saturating_sub(1);

        // Allocate temporary padded buffer
        let data_type: DataType = self.config.in_projection_config.activation_precision().into();
        let element_size = data_type.size_in_bytes();
        let padded_rows = state_stride + suffix_length;
        let padded_size = padded_rows * self.model_dim * element_size;
        let mut padded_buf = state.context().create_buffer(padded_size).expect("Failed to create padded buffer");
        self.short_conv_pack.encode(
            conv_state.buffer().borrow().deref(),
            in_proj.buffer().borrow().deref(),
            &mut padded_buf,
            state_stride as u32,
            suffix_length as u32,
            self.model_dim as u32 * 3,
            self.model_dim as u32,
            encoder,
        );

        self.short_conv_prefill.encode(
            &padded_buf,
            in_proj.buffer().borrow().deref(),
            self.conv_weight.buffer().borrow().deref(),
            bias_buf_borrow.as_deref(),
            out.buffer().borrow_mut().deref_mut(),
            conv_state.buffer().borrow_mut().deref_mut(),
            suffix_length as u32,
            kernel_size as u32,
            self.model_dim as u32 * 3,
            state_stride as u32,
            self.model_dim as u32,
            encoder,
        )
    }

    fn run_trie_conv(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
        sampling_start: usize,
        trie_len: usize,
    ) {
        if self.model_dim == 0 || trie_len == 0 {
            return;
        }

        let in_proj = state.array(ArrayId::SsmInProj);
        let parents = state.array(ArrayId::TokenParents);
        let conv_state = state.array(ArrayId::ShortConvState(self.layer_index));
        let suffix_state = state.array(ArrayId::ShortConvSuffixState(self.layer_index));
        let out = state.array(ArrayId::AttentionOutput);

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
        let trie_bias_buf_rc = self.conv_bias.as_ref().map(|b| b.buffer());
        let trie_bias_buf_borrow = trie_bias_buf_rc.as_ref().map(|rc| rc.borrow());

        self.short_conv_trie.encode(
            (in_proj.buffer().borrow().deref(), in_proj_offset),
            self.conv_weight.buffer().borrow().deref(),
            trie_bias_buf_borrow.as_deref(),
            (conv_state.buffer().borrow().deref(), base_state_offset),
            (parents.buffer().borrow().deref(), parents_offset),
            (out.buffer().borrow_mut().deref_mut(), out_offset),
            (suffix_state.buffer().borrow_mut().deref_mut(), suffix_state_offset),
            trie_len as u32,
            kernel_size as u32,
            in_proj_stride as u32,
            state_stride as u32,
            self.model_dim as u32,
            encoder,
        );
    }

    fn run_decode_conv(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
        suffix_length: usize,
    ) {
        if self.model_dim == 0 || suffix_length == 0 {
            return;
        }

        let in_proj = state.array(ArrayId::SsmInProj);
        let conv_state = state.array(ArrayId::ShortConvState(self.layer_index));
        let out = state.array(ArrayId::AttentionOutput);

        let decode_bias_buf_rc = self.conv_bias.as_ref().map(|b| b.buffer());
        let decode_bias_buf_borrow = decode_bias_buf_rc.as_ref().map(|rc| rc.borrow());

        let kernel_size = self.config.kernel_size;
        let state_stride = kernel_size.saturating_sub(1);

        self.short_conv_decode.encode(
            in_proj.buffer().borrow().deref(),
            self.conv_weight.buffer().borrow().deref(),
            decode_bias_buf_borrow.as_deref(),
            None::<&B::Buffer>,
            out.buffer().borrow_mut().deref_mut(),
            conv_state.buffer().borrow_mut().deref_mut(),
            suffix_length as u32,
            kernel_size as u32,
            self.model_dim as u32 * 3,
            state_stride as u32,
            self.model_dim as u32,
            encoder,
        )
    }

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let active_suffix_length = state.active_suffix_length();
        if active_suffix_length == 0 {
            return Ok(());
        }

        self.in_projection.encode(state, encoder)?;

        self.run_conv(state, encoder, active_suffix_length);

        self.out_projection.encode(state, encoder)?;

        Ok(())
    }
}
