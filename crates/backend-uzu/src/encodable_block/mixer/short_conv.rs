use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, AllocationType, Backend, Context, Encoder, Kernels,
        kernel::{ShortConvDecodeKernel, ShortConvPackKernel, ShortConvPrefillKernel, ShortConvTrieKernel},
    },
    config::token_mixer::short_conv::ShortConvConfig,
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::{Linear, LinearBlockError},
        mixer::{Mixer, MixerState, attention::rope::PrecalculatedRoPE},
    },
    parameters::{ParameterLoaderError, ParameterTree},
    utils::maybe_mut::MaybeMut,
};

enum ShortConvStateSuffixStatus<B: Backend> {
    Flat {
        suffix_length: usize,
    },
    Trie {
        conv_states: Allocation<B>,
    },
}

pub struct ShortConvState<B: Backend> {
    conv_state: Allocation<B>,
    suffix_state: Option<ShortConvStateSuffixStatus<B>>,
}

impl<B: Backend> MixerState<B> for ShortConvState<B> {
    fn prepare(
        &mut self,
        _context_length: usize,
        _suffix_length: usize,
        _context: &B::Context,
    ) -> Result<(), B::Error> {
        Ok(())
    }

    fn encode_accept(
        &mut self,
        accepted_indices: &[usize],
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let suffix_state_type =
            self.suffix_state.take().expect("Called short conv state encode accept on a state with nothing to accept");

        let accepted_index = *accepted_indices.last().expect("short conv state attempted to accept zero indicies");

        match suffix_state_type {
            ShortConvStateSuffixStatus::Flat {
                suffix_length,
            } => {
                assert!(accepted_index == suffix_length - 1, "attempted to do a partial flat short conv state accept");
                Ok(())
            },
            ShortConvStateSuffixStatus::Trie {
                conv_states,
            } => {
                let conv_state_size = self.conv_state.size();
                encoder.encode_copy(
                    &conv_states,
                    (accepted_index * conv_state_size)..((accepted_index + 1) * conv_state_size),
                    &mut self.conv_state,
                    ..,
                );
                Ok(())
            },
        }
    }
}

pub struct ShortConv<B: Backend> {
    hidden_dim: usize,
    data_type: DataType,
    kernel_size: usize,
    in_projection: Box<dyn Linear<B>>,
    out_projection: Box<dyn Linear<B>>,
    short_conv_pack: <B::Kernels as Kernels>::ShortConvPackKernel,
    short_conv_prefill: <B::Kernels as Kernels>::ShortConvPrefillKernel,
    short_conv_decode: <B::Kernels as Kernels>::ShortConvDecodeKernel,
    short_conv_trie: <B::Kernels as Kernels>::ShortConvTrieKernel,
    conv_weight: Allocation<B>,
    conv_bias: Option<Allocation<B>>,
}

#[derive(Debug, Error)]
pub enum ShortConvNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Unsupported configuration: {0}")]
    UnsupportedConfiguration(String),
}

impl<B: Backend> ShortConv<B> {
    pub fn new(
        hidden_dim: usize,
        data_type: DataType,
        config: &ShortConvConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<(Self, Option<Allocation<B>>), ShortConvNewError<B>> {
        let kernel_size = config.kernel_size;
        if kernel_size < 2 {
            return Err(ShortConvNewError::UnsupportedConfiguration(format!(
                "kernel_size must be >= 2, got {}",
                kernel_size
            )));
        }

        let (in_projection, in_projection_input_hadamard_factors) = <dyn Linear<B>>::new_extracting_input_hadamard(
            hidden_dim,
            [hidden_dim * 3],
            false,
            context,
            data_type,
            &parameter_tree.subtree("in_projection")?,
        )?;

        let out_projection = <dyn Linear<B>>::new(
            hidden_dim,
            [hidden_dim],
            false,
            context,
            data_type,
            &parameter_tree.subtree("out_projection")?,
        )?;

        let conv_tree = parameter_tree.subtree("conv")?;

        let conv_data_type = DataType::F32;
        let conv_weight =
            conv_tree.leaf("weights")?.validate(&[hidden_dim, kernel_size], conv_data_type)?.read_allocation()?;

        let has_bias = config.conv_config.has_biases;
        let conv_bias = if has_bias {
            Some(conv_tree.leaf("biases")?.validate(&[hidden_dim], conv_data_type)?.read_allocation()?)
        } else {
            None
        };

        let short_conv_pack = <B::Kernels as Kernels>::ShortConvPackKernel::new(context, data_type)
            .map_err(ShortConvNewError::Backend)?;
        let short_conv_prefill =
            <B::Kernels as Kernels>::ShortConvPrefillKernel::new(context, data_type, conv_data_type, has_bias)
                .map_err(ShortConvNewError::Backend)?;
        let short_conv_decode =
            <B::Kernels as Kernels>::ShortConvDecodeKernel::new(context, data_type, conv_data_type, has_bias, true)
                .map_err(ShortConvNewError::Backend)?;
        let short_conv_trie =
            <B::Kernels as Kernels>::ShortConvTrieKernel::new(context, data_type, conv_data_type, has_bias)
                .map_err(ShortConvNewError::Backend)?;

        Ok((
            Self {
                hidden_dim,
                data_type,
                kernel_size,
                in_projection,
                out_projection,
                short_conv_pack,
                short_conv_prefill,
                short_conv_decode,
                short_conv_trie,
                conv_weight,
                conv_bias,
            },
            in_projection_input_hadamard_factors,
        ))
    }

    fn encode_decode_conv(
        &self,
        in_projected: &Allocation<B>,
        state: &mut ShortConvState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut conv_output = encoder.allocate_scratch(size_for_shape(&[self.hidden_dim], self.data_type))?;
        self.short_conv_decode.encode(
            in_projected,
            &self.conv_weight,
            self.conv_bias.as_ref(),
            None::<&Allocation<B>>,
            &mut conv_output,
            &mut state.conv_state,
            1,
            self.kernel_size as u32,
            self.hidden_dim as u32 * 3,
            (self.kernel_size - 1) as u32,
            self.hidden_dim as u32,
            encoder,
        );
        Ok(conv_output)
    }

    fn encode_prefill_conv(
        &self,
        in_projected: &Allocation<B>,
        batch_dim: usize,
        state: &mut ShortConvState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let state_stride = self.kernel_size - 1;
        let padded_rows = state_stride + batch_dim;

        let mut padded = encoder.allocate_scratch(size_for_shape(&[padded_rows, self.hidden_dim], self.data_type))?;
        self.short_conv_pack.encode(
            &state.conv_state,
            in_projected,
            &mut padded,
            state_stride as u32,
            batch_dim as u32,
            self.hidden_dim as u32 * 3,
            self.hidden_dim as u32,
            encoder,
        );

        let mut conv_output =
            encoder.allocate_scratch(size_for_shape(&[batch_dim, self.hidden_dim], self.data_type))?;
        self.short_conv_prefill.encode(
            &padded,
            in_projected,
            &self.conv_weight,
            self.conv_bias.as_ref(),
            &mut conv_output,
            &mut state.conv_state,
            batch_dim as u32,
            self.kernel_size as u32,
            self.hidden_dim as u32 * 3,
            state_stride as u32,
            self.hidden_dim as u32,
            encoder,
        );
        Ok(conv_output)
    }

    fn encode_trie_conv(
        &self,
        in_projected: &Allocation<B>,
        batch_dim: usize,
        token_parents: &Allocation<B>,
        state: &mut ShortConvState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(Allocation<B>, Allocation<B>), B::Error> {
        let mut conv_output =
            encoder.allocate_scratch(size_for_shape(&[batch_dim, self.hidden_dim], self.data_type))?;
        let mut conv_states = encoder
            .allocate_scratch(size_for_shape(&[batch_dim, self.kernel_size - 1, self.hidden_dim], self.data_type))?;
        self.short_conv_trie.encode(
            in_projected,
            &self.conv_weight,
            self.conv_bias.as_ref(),
            &state.conv_state,
            token_parents,
            &mut conv_output,
            &mut conv_states,
            batch_dim as u32,
            self.kernel_size as u32,
            (self.hidden_dim * 3) as u32,
            (self.kernel_size - 1) as u32,
            self.hidden_dim as u32,
            encoder,
        );
        Ok((conv_output, conv_states))
    }
}

impl<B: Backend> Mixer<B> for ShortConv<B> {
    fn speculation_supported(&self) -> bool {
        true
    }

    fn max_context_length(&self) -> Option<usize> {
        None
    }

    fn create_empty_state(
        &self,
        _max_context_length: Option<usize>,
        context: &B::Context,
    ) -> Result<Box<dyn MixerState<B>>, B::Error> {
        let mut conv_state = context.create_allocation(
            size_for_shape(&[self.kernel_size - 1, self.hidden_dim], self.data_type),
            AllocationType::Global,
        )?;

        let suffix_capacity = 1024; // TODO: remove hardcoded suffix capacity
        let mut suffix_state = context.create_allocation(
            size_for_shape(&[suffix_capacity, self.kernel_size - 1, self.hidden_dim], self.data_type),
            AllocationType::Global,
        )?;

        let mut zero_encoder = Encoder::<B>::new(context)?;
        zero_encoder.encode_fill(&mut conv_state, 0);
        zero_encoder.encode_fill(&mut suffix_state, 0);
        zero_encoder.end_encoding().submit().wait_until_completed()?;

        Ok(Box::new(ShortConvState {
            conv_state,
            suffix_state: None,
        }))
    }

    fn encode(
        &self,
        hidden: Allocation<B>,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        assert!(precalculated_rope.is_none(), "unexpected rope for short conv mixer");

        let state = state.expect("short conv requires state");
        let state = state.downcast::<ShortConvState<B>>().expect("incorrect type of short conv state");
        let MaybeMut::Mut(state) = state else {
            panic!("incorrect state access for short conv state");
        };

        assert!(state.suffix_state.is_none(), "short conv called with state with unaccepted tokens");

        let in_projected = self.in_projection.encode(hidden, batch_dim.size(), encoder)?;

        let conv_output = if batch_dim.full_accept() {
            let conv_output = if batch_dim.size() == 1 {
                self.encode_decode_conv(&in_projected, state, encoder)?
            } else {
                self.encode_prefill_conv(&in_projected, batch_dim.size(), state, encoder)?
            };
            state.suffix_state = Some(ShortConvStateSuffixStatus::Flat {
                suffix_length: batch_dim.size(),
            });
            conv_output
        } else {
            let mut token_parents = encoder.allocate_constant(batch_dim.size() * DataType::I32.size_in_bytes())?;
            token_parents.copyin(batch_dim.parents());
            let (conv_output, conv_states) =
                self.encode_trie_conv(&in_projected, batch_dim.size(), &token_parents, state, encoder)?;
            state.suffix_state = Some(ShortConvStateSuffixStatus::Trie {
                conv_states,
            });
            conv_output
        };

        self.out_projection.encode(conv_output, batch_dim.size(), encoder)
    }
}
