use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        kernel::{ShortConvDecodeKernel, ShortConvPackKernel, ShortConvPrefillKernel, ShortConvTrieKernel},
    },
    config::token_mixer::short_conv::ShortConvConfig,
    data_type::DataType,
    encodable_block::linear::{Linear, LinearBlockError},
    forward_pass::short_conv_layer::ShortConvLayer,
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum ShortConvMixerError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Unsupported configuration: {0}")]
    UnsupportedConfiguration(String),
    #[error("Linear error: {0}")]
    LinearError(#[from] LinearBlockError<B>),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
}

pub struct ShortConvMixer<B: Backend> {
    model_dim: usize,
    kernel_size: usize,
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

pub struct ShortConvArguments<'a, B: Backend> {
    pub active_row_count: usize,
    pub sampling_start: usize,
    pub sampling_length: usize,
    pub token_parents: &'a Allocation<B>,
    pub layer: &'a mut ShortConvLayer<B>,
}

impl<B: Backend> ShortConvMixer<B> {
    pub fn new(
        context: &B::Context,
        short_conv_config: &ShortConvConfig,
        model_dim: usize,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<(Self, Option<Allocation<B>>), ShortConvMixerError<B>> {
        if short_conv_config.kernel_size < 2 {
            return Err(ShortConvMixerError::UnsupportedConfiguration(format!(
                "kernel_size must be >= 2, got {}",
                short_conv_config.kernel_size
            )));
        }

        let conv_tree = parameter_tree.subtree("conv")?;

        let (in_projection, in_proj_input_hadamard_factors) = <dyn Linear<B>>::new_extracting_input_hadamard(
            model_dim,
            [model_dim * 3],
            false,
            context,
            data_type,
            &parameter_tree.subtree("in_projection")?,
        )?;

        let out_projection = <dyn Linear<B>>::new(
            model_dim,
            [model_dim],
            false,
            context,
            data_type,
            &parameter_tree.subtree("out_projection")?,
        )?;

        let conv_data_type = DataType::F32;
        let conv_weight = conv_tree
            .leaf("weights")?
            .validate(&[model_dim, short_conv_config.kernel_size], conv_data_type)?
            .read_allocation()?;
        let conv_bias = if short_conv_config.conv_config.has_biases {
            Some(conv_tree.leaf("biases")?.validate(&[model_dim], conv_data_type)?.read_allocation()?)
        } else {
            None
        };

        let has_bias = short_conv_config.conv_config.has_biases;
        let short_conv_pack = <B::Kernels as Kernels>::ShortConvPackKernel::new(context, data_type)
            .map_err(ShortConvMixerError::BackendError)?;
        let short_conv_prefill =
            <B::Kernels as Kernels>::ShortConvPrefillKernel::new(context, data_type, conv_data_type, has_bias)
                .map_err(ShortConvMixerError::BackendError)?;
        let short_conv_decode =
            <B::Kernels as Kernels>::ShortConvDecodeKernel::new(context, data_type, conv_data_type, has_bias, true)
                .map_err(ShortConvMixerError::BackendError)?;
        let short_conv_trie =
            <B::Kernels as Kernels>::ShortConvTrieKernel::new(context, data_type, conv_data_type, has_bias)
                .map_err(ShortConvMixerError::BackendError)?;

        Ok((
            Self {
                model_dim,
                kernel_size: short_conv_config.kernel_size,
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
        ))
    }

    fn run_conv(
        &self,
        layer: &mut ShortConvLayer<B>,
        token_parents: &Allocation<B>,
        sampling_start: usize,
        sampling_length: usize,
        in_proj: &Allocation<B>,
        encoder: &mut Encoder<B>,
        active_row_count: usize,
    ) -> Result<Allocation<B>, B::Error> {
        layer.clear_suffix_state_valid_range();
        let mut out = encoder.allocate_scratch(size_for_shape(&[active_row_count, self.model_dim], self.data_type))?;

        if active_row_count == 1 {
            self.run_decode_conv(layer, in_proj, &mut out, encoder, 1)?;
            return Ok(out);
        }

        if sampling_length == 0 {
            self.run_prefill_conv(layer, in_proj, &mut out, encoder, active_row_count)?;
            return Ok(out);
        }

        let trie_len = sampling_length;

        if trie_len <= 1 {
            self.run_prefill_conv(layer, in_proj, &mut out, encoder, active_row_count)?;
            return Ok(out);
        }

        if sampling_start > 0 {
            if sampling_start == 1 {
                self.run_decode_conv(layer, in_proj, &mut out, encoder, 1)?;
            } else {
                self.run_prefill_conv(layer, in_proj, &mut out, encoder, sampling_start)?;
            }
        }

        self.run_trie_conv(layer, token_parents, in_proj, &mut out, encoder, sampling_start, trie_len)?;
        layer.set_suffix_state_valid_range(sampling_start, trie_len);
        Ok(out)
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

        let kernel_size = self.kernel_size;
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

        let elem_bytes = self.data_type.size_in_bytes();

        let kernel_size = self.kernel_size;
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

        let kernel_size = self.kernel_size;
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
        let conv_output =
            self.run_conv(layer, token_parents, sampling_start, sampling_length, &in_proj, encoder, active_row_count)?;

        self.out_projection.encode(conv_output, active_row_count, encoder)
    }
}
