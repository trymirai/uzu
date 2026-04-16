use crate::{
    DataType,
    array::{Array, size_for_shape},
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        kernel::{ShortConvDecodeKernel, ShortConvPackKernel, ShortConvPrefillKernel, ShortConvTrieKernel},
    },
    config::{DecoderLayerType, ShortConvConfig},
    encodable_block::linear::Linear,
    forward_pass::short_conv_layer::ShortConvLayer,
    parameters::{ParameterTree, resolve_subtree},
};

pub struct ShortConvMixer<B: Backend> {
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
    data_type: DataType,
}

pub(crate) struct ShortConvArguments<'a, B: Backend> {
    pub context: &'a B::Context,
    pub active_row_count: usize,
    pub sampling_start: usize,
    pub sampling_length: usize,
    pub token_parents: &'a Allocation<B>,
    pub layer: &'a mut ShortConvLayer<B>,
}

impl<B: Backend> ShortConvMixer<B> {
    pub fn new(
        context: &B::Context,
        layer_type: DecoderLayerType,
        short_conv_config: ShortConvConfig,
        layer_index: usize,
        model_dim: usize,
        decoder_layer_loader: &ParameterTree<B::Context>,
    ) -> Self {
        if !matches!(layer_type, DecoderLayerType::ShortConv { .. }) {
            panic!("Layer {} marked as non-ShortConv but ShortConv config provided", layer_index);
        }

        let mixer_tree = resolve_subtree(decoder_layer_loader, &["mixer"]);
        let conv_tree = resolve_subtree(&mixer_tree, &["conv"]);

        let data_type: DataType = short_conv_config.in_projection_config.activation_precision().into();

        let in_projection = <dyn Linear<B>>::new(
            &short_conv_config.in_projection_config,
            false,
            model_dim,
            [model_dim * 3],
            context,
            &resolve_subtree(&mixer_tree, &["in_projection", "in_proj"]),
        )
        .expect("Failed to create in-projection kernel");

        let out_projection = <dyn Linear<B>>::new(
            &short_conv_config.out_projection_config,
            false,
            model_dim,
            [model_dim],
            context,
            &resolve_subtree(&mixer_tree, &["out_projection", "out_proj"]),
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

        Self {
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
            data_type,
        }
    }

    fn clear_suffix_state_valid_range(
        &self,
        layer: &ShortConvLayer<B>,
    ) {
        layer.clear_suffix_state_valid_range();
    }

    fn set_suffix_state_valid_range(
        &self,
        layer: &ShortConvLayer<B>,
        start: usize,
        len: usize,
    ) {
        layer.set_suffix_state_valid_range(start, len);
    }

    fn run_conv(
        &self,
        layer: &mut ShortConvLayer<B>,
        token_parents: &Allocation<B>,
        sampling_start: usize,
        sampling_length: usize,
        in_proj: &Allocation<B>,
        out: &mut Allocation<B>,
        encoder: &mut Encoder<B>,
        active_row_count: usize,
    ) -> Result<(), B::Error> {
        self.clear_suffix_state_valid_range(layer);

        if active_row_count == 1 {
            self.run_decode_conv(layer, in_proj, out, encoder, 1);
            return Ok(());
        }

        if sampling_length == 0 {
            return self.run_prefill_conv(layer, in_proj, out, encoder, active_row_count);
        }

        let trie_len = sampling_length;

        if trie_len <= 1 {
            return self.run_prefill_conv(layer, in_proj, out, encoder, active_row_count);
        }

        if sampling_start > 0 {
            if sampling_start == 1 {
                self.run_decode_conv(layer, in_proj, out, encoder, 1);
            } else {
                self.run_prefill_conv(layer, in_proj, out, encoder, sampling_start)?;
            }
        }

        self.run_trie_conv(layer, token_parents, in_proj, out, encoder, sampling_start, trie_len);
        self.set_suffix_state_valid_range(layer, sampling_start, trie_len);
        Ok(())
    }

    fn run_prefill_conv(
        &self,
        layer: &mut ShortConvLayer<B>,
        in_proj: &Allocation<B>,
        out: &mut Allocation<B>,
        encoder: &mut Encoder<B>,
        suffix_length: usize,
    ) -> Result<(), B::Error> {
        if self.model_dim == 0 || suffix_length == 0 {
            return Ok(());
        }

        let bias_buf_rc = self.conv_bias.as_ref().map(|b| b.buffer());
        let bias_buf_borrow = bias_buf_rc.as_ref().map(|rc| rc.borrow());
        let conv_weight = self.conv_weight.buffer();
        let conv_weight = conv_weight.borrow();

        let kernel_size = self.config.kernel_size;
        let state_stride = kernel_size.saturating_sub(1);

        let padded_rows = state_stride + suffix_length;
        let mut padded = encoder.allocate_scratch(size_for_shape(&[padded_rows, self.model_dim], self.data_type))?;
        self.short_conv_pack.encode(
            &layer.conv_state,
            in_proj,
            &mut padded,
            state_stride as u32,
            suffix_length as u32,
            self.model_dim as u32 * 3,
            self.model_dim as u32,
            encoder,
        );
        self.short_conv_prefill.encode(
            &padded,
            in_proj,
            &*conv_weight,
            bias_buf_borrow.as_deref(),
            out,
            &mut layer.conv_state,
            suffix_length as u32,
            kernel_size as u32,
            self.model_dim as u32 * 3,
            state_stride as u32,
            self.model_dim as u32,
            encoder,
        );
        Ok(())
    }

    fn run_trie_conv(
        &self,
        layer: &mut ShortConvLayer<B>,
        token_parents: &Allocation<B>,
        in_proj: &Allocation<B>,
        out: &mut Allocation<B>,
        encoder: &mut Encoder<B>,
        sampling_start: usize,
        trie_len: usize,
    ) {
        if self.model_dim == 0 || trie_len == 0 {
            return;
        }

        let elem_bytes = DataType::from(self.config.in_projection_config.activation_precision()).size_in_bytes();

        let kernel_size = self.config.kernel_size;
        let state_stride = kernel_size.saturating_sub(1);
        let in_proj_stride = self.model_dim * 3;

        let in_proj_offset = sampling_start * in_proj_stride * elem_bytes;
        let out_offset = sampling_start * self.model_dim * elem_bytes;
        let suffix_state_offset = sampling_start * self.model_dim * state_stride * elem_bytes;
        let base_state_offset = 0;
        let parents_offset = sampling_start * std::mem::size_of::<i32>();
        let trie_bias_buf_rc = self.conv_bias.as_ref().map(|b| b.buffer());
        let trie_bias_buf_borrow = trie_bias_buf_rc.as_ref().map(|rc| rc.borrow());
        let conv_weight = self.conv_weight.buffer();
        let conv_weight = conv_weight.borrow();

        self.short_conv_trie.encode(
            (in_proj, in_proj_offset),
            &*conv_weight,
            trie_bias_buf_borrow.as_deref(),
            (&layer.conv_state, base_state_offset),
            (token_parents, parents_offset),
            (out, out_offset),
            (&mut layer.suffix_state, suffix_state_offset),
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
        layer: &mut ShortConvLayer<B>,
        in_proj: &Allocation<B>,
        out: &mut Allocation<B>,
        encoder: &mut Encoder<B>,
        suffix_length: usize,
    ) {
        if self.model_dim == 0 || suffix_length == 0 {
            return;
        }

        let decode_bias_buf_rc = self.conv_bias.as_ref().map(|b| b.buffer());
        let decode_bias_buf_borrow = decode_bias_buf_rc.as_ref().map(|rc| rc.borrow());
        let conv_weight = self.conv_weight.buffer();
        let conv_weight = conv_weight.borrow();

        let kernel_size = self.config.kernel_size;
        let state_stride = kernel_size.saturating_sub(1);

        self.short_conv_decode.encode(
            in_proj,
            &*conv_weight,
            decode_bias_buf_borrow.as_deref(),
            None::<&Allocation<B>>,
            out,
            &mut layer.conv_state,
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
        args: ShortConvArguments<'_, B>,
        input: &Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let ShortConvArguments {
            context,
            active_row_count,
            sampling_start,
            sampling_length,
            token_parents,
            layer,
        } = args;
        assert!(active_row_count > 0, "ShortConv mixer requires at least one active row");

        let in_proj = self.in_projection.encode(context, input, active_row_count, encoder)?;
        let mut conv_output =
            encoder.allocate_scratch(size_for_shape(&[active_row_count, self.model_dim], self.data_type))?;

        self.run_conv(
            layer,
            token_parents,
            sampling_start,
            sampling_length,
            &in_proj,
            &mut conv_output,
            encoder,
            active_row_count,
        )?;

        self.out_projection.encode(context, &conv_output, active_row_count, encoder)
    }
}
