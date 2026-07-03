use std::ops::Range;

use thiserror::Error;

use crate::{
    backends::common::{Allocation, Backend, Encoder},
    config::{rope::AnyRoPEConfig, transformer::TransformerConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        mixer::{MixerState, attention::rope::PrecalculatedRoPE},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar},
        transformer_layer::{TransformerLayer, TransformerLayerError},
    },
    parameters::{ParameterLoaderError, ParameterTree},
    utils::maybe_mut::MaybeMut,
};

enum TransformerLayerStateType<B: Backend> {
    Owned(Box<dyn MixerState<B>>),
    Shared(usize),
}

pub struct TransformerState<B: Backend> {
    layer_states: Box<[TransformerLayerStateType<B>]>,
    context_length: usize,
}

impl<B: Backend> TransformerState<B> {
    pub fn context_length(&self) -> usize {
        self.context_length
    }

    pub fn prepare(
        &mut self,
        context_length: usize,
        suffix_length: usize,
        context: &B::Context,
    ) -> Result<(), B::Error> {
        for layer_state in &mut self.layer_states {
            let TransformerLayerStateType::Owned(layer_state) = layer_state else {
                continue;
            };

            layer_state.prepare(context_length, suffix_length, context)?;
        }

        Ok(())
    }

    pub fn encode_accept(
        &mut self,
        accepted_indices: &[usize],
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        for layer_state in &mut self.layer_states {
            let TransformerLayerStateType::Owned(layer_state) = layer_state else {
                continue;
            };

            layer_state.encode_accept(accepted_indices, encoder)?;
        }

        self.context_length += accepted_indices.len();

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum TransformerNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Layer error: {0}")]
    Layer(#[from] TransformerLayerError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
}

pub struct Transformer<B: Backend> {
    ropes: Box<[AnyRoPEConfig]>,
    layers: Box<[(TransformerLayer<B>, Option<usize>)]>,
    output_norm: Normalization<B>,
}

impl<B: Backend> Transformer<B> {
    pub fn new(
        context: &B::Context,
        output_norm_hadamard_factors: Option<Allocation<B>>,
        data_type: DataType,
        transformer_config: &TransformerConfig,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Self, TransformerNewError<B>> {
        let mut ropes: Vec<AnyRoPEConfig> = Vec::new();

        let layers = transformer_config
            .layer_configs
            .iter()
            .enumerate()
            .map(|(layer_index, layer_config)| {
                let layer_loader = parameter_tree.subtree(&format!("layers.{}", layer_index))?;

                let rope = layer_config.rope_config.as_ref().map(|layer_rope_config| {
                    ropes.iter().position(|rope_config| rope_config == layer_rope_config).unwrap_or_else(|| {
                        ropes.push(layer_rope_config.clone());
                        ropes.len() - 1
                    })
                });

                let layer = TransformerLayer::new(
                    context,
                    transformer_config,
                    layer_config,
                    layer_index,
                    &layer_loader,
                    data_type,
                )?;

                Ok((layer, rope))
            })
            .collect::<Result<Box<[_]>, TransformerNewError<B>>>()?;

        let output_norm = Normalization::new(
            transformer_config.model_dim,
            output_norm_hadamard_factors,
            true,
            true,
            PostLayerScalar::None,
            data_type,
            &transformer_config.output_norm_config,
            &parameter_tree.subtree("output_norm")?,
            context,
        )?;

        Ok(Self {
            ropes: ropes.into_boxed_slice(),
            layers,
            output_norm,
        })
    }

    pub fn speculation_supported(&self) -> bool {
        self.layers.iter().all(|(layer, _rope)| layer.mixer.speculation_supported())
    }

    pub fn max_context_length(&self) -> Option<usize> {
        self.layers.iter().map(|(layer, _rope_index)| layer.mixer.max_context_length()).fold(None, |acc, el| {
            match (acc, el) {
                (Some(a), Some(b)) => Some(usize::min(a, b)),
                (Some(x), None) | (None, Some(x)) => Some(x),
                (None, None) => None,
            }
        })
    }

    pub fn create_empty_state(
        &self,
        max_context_length: Option<usize>,
        context: &B::Context,
    ) -> Result<TransformerState<B>, B::Error> {
        let layer_states = self
            .layers
            .iter()
            .map(|(layer, _rope)| match layer.kv_source_layer_index {
                None => {
                    layer.mixer.create_empty_state(max_context_length, context).map(TransformerLayerStateType::Owned)
                },
                Some(kv_source_layer_index) => Ok(TransformerLayerStateType::Shared(kv_source_layer_index)),
            })
            .collect::<Result<_, B::Error>>()?;

        let context_length = 0;

        Ok(TransformerState {
            layer_states,
            context_length,
        })
    }

    pub fn encode(
        &self,
        input: Allocation<B>,
        per_layer_inputs: Option<&Allocation<B>>,
        batch_dim: &BatchTopology,
        output_range: Option<Range<usize>>,
        mut state: Option<&mut TransformerState<B>>,
        encoder: &mut Encoder<B>,
    ) -> Result<Option<Allocation<B>>, B::Error> {
        let mut hidden = input;

        let mut shortcut = encoder.allocate_scratch(hidden.size())?;

        let context_length = state.as_ref().map(|state| state.context_length).unwrap_or(0);
        let token_positions =
            batch_dim.heights().map(|rel_pos| context_length + rel_pos as usize).collect::<Box<[usize]>>();

        let precalculated_ropes = self
            .ropes
            .iter()
            .map(|rope_config| PrecalculatedRoPE::precalculate(rope_config, &token_positions, encoder))
            .collect::<Result<Box<[_]>, B::Error>>()?;

        for (layer, layer_rope_index) in self.layers.iter() {
            let precalculated_rope = layer_rope_index.map(|i| &precalculated_ropes[i]);

            let layer_state = if let Some(state) = &mut state {
                Some(match &mut state.layer_states[layer.layer_index] {
                    TransformerLayerStateType::Owned(layer_state) => MaybeMut::Mut(layer_state.as_mut()),
                    TransformerLayerStateType::Shared(owned_layer_index) => {
                        let TransformerLayerStateType::Owned(owned_layer) = &state.layer_states[*owned_layer_index]
                        else {
                            panic!("shared layer doesn't point to an owned layer");
                        };
                        MaybeMut::Const(owned_layer.as_ref())
                    },
                })
            } else {
                None
            };

            hidden = layer.encode(
                hidden,
                &mut shortcut,
                per_layer_inputs,
                precalculated_rope,
                batch_dim,
                layer_state,
                encoder,
            )?;
        }

        let Some(output_range) = output_range else {
            return Ok(None);
        };

        let output_normalized =
            self.output_norm.encode(&hidden, output_range.start, output_range.len(), Some(&mut shortcut), encoder)?;

        Ok(Some(output_normalized))
    }
}
