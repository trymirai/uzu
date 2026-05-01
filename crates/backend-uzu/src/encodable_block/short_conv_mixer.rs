use crate::{
    DataType,
    array::size_for_shape,
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
    conv_weight: Allocation<B>,
    conv_bias: Option<Allocation<B>>,
    data_type: DataType,
}

pub(crate) struct ShortConvArguments<'a, B: Backend> {
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
    ) -> (Self, Option<Allocation<B>>) {
        if !matches!(layer_type, DecoderLayerType::ShortConv { .. }) {
            panic!("Layer {} marked as non-ShortConv but ShortConv config provided", layer_index);
        }
        assert!(
            short_conv_config.kernel_size >= 2,
            "ShortConv kernel_size must be >= 2, got {}",
            short_conv_config.kernel_size
        );

        let mixer_tree = resolve_subtree(decoder_layer_loader, &["mixer"]);
        let conv_tree = resolve_subtree(&mixer_tree, &["conv"]);

        let data_type: DataType = short_conv_config.in_projection_config.activation_precision().into();

        let (in_projection, in_proj_input_hadamard_factors) = <dyn Linear<B>>::new_extracting_input_hadamard(
            &short_conv_config.in_projection_config,
            model_dim,
            [model_dim * 3],
            context,
            &resolve_subtree(&mixer_tree, &["in_projection", "in_proj"]),
        )
        .expect("Failed to create in-projection kernel");

        let out_projection = <dyn Linear<B>>::new(
            &short_conv_config.out_projection_config,
            model_dim,
            [model_dim],
            context,
            &resolve_subtree(&mixer_tree, &["out_projection", "out_proj"]),
        )
        .expect("Failed to create out-projection kernel");

        let conv_weight = conv_tree.leaf("weights").unwrap().read_allocation().unwrap();
        let conv_bias = if short_conv_config.conv_config.has_biases {
            Some(conv_tree.leaf("biases").unwrap().read_allocation().unwrap())
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
            },
            in_proj_input_hadamard_factors,
        )
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
        layer.clear_suffix_state_valid_range();

        if active_row_count == 1 {
            self.run_decode_conv(layer, in_proj, out, encoder, 1)?;
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
                self.run_decode_conv(layer, in_proj, out, encoder, 1)?;
            } else {
                self.run_prefill_conv(layer, in_proj, out, encoder, sampling_start)?;
            }
        }

        self.run_trie_conv(layer, token_parents, in_proj, out, encoder, sampling_start, trie_len)?;
        layer.set_suffix_state_valid_range(sampling_start, trie_len);
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

        let kernel_size = self.config.kernel_size;
        let state_stride = kernel_size - 1;

        let padded_rows = state_stride + suffix_length;
        let mut padded = encoder.allocate_scratch(size_for_shape(&[padded_rows, self.model_dim], self.data_type))?;
        self.short_conv_pack.encode(
            (&layer.conv_state, 0),
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
            &self.conv_weight,
            self.conv_bias.as_ref(),
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
    ) -> Result<(), B::Error> {
        if self.model_dim == 0 || trie_len == 0 {
            return Ok(());
        }

        let elem_bytes = DataType::from(self.config.in_projection_config.activation_precision()).size_in_bytes();

        let kernel_size = self.config.kernel_size;
        let state_stride = kernel_size - 1;
        let in_proj_stride = self.model_dim * 3;

        let in_proj_view = (in_proj, sampling_start * in_proj_stride * elem_bytes);
        let parents_view = (token_parents, sampling_start * std::mem::size_of::<i32>());
        let out_view = (&mut *out, sampling_start * self.model_dim * elem_bytes);
        self.short_conv_trie.encode(
            in_proj_view,
            &self.conv_weight,
            self.conv_bias.as_ref(),
            (&layer.conv_state, 0),
            parents_view,
            out_view,
            (&mut layer.suffix_state, sampling_start * self.model_dim * state_stride * elem_bytes),
            trie_len as u32,
            kernel_size as u32,
            in_proj_stride as u32,
            state_stride as u32,
            self.model_dim as u32,
            encoder,
        );
        Ok(())
    }

    fn run_decode_conv(
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

        let kernel_size = self.config.kernel_size;
        let state_stride = kernel_size - 1;

        self.short_conv_decode.encode(
            in_proj,
            &self.conv_weight,
            self.conv_bias.as_ref(),
            None::<&Allocation<B>>,
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

    pub fn encode(
        &self,
        args: ShortConvArguments<B>,
        input: Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let ShortConvArguments {
            active_row_count,
            sampling_start,
            sampling_length,
            token_parents,
            layer,
        } = args;
        assert!(active_row_count > 0, "ShortConv mixer requires at least one active row");

        let in_proj = self.in_projection.encode(input, active_row_count, encoder)?;
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

        self.out_projection.encode(conv_output, active_row_count, encoder)
    }
}
