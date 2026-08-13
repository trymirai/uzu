use std::ops::Range;

use thiserror::Error;

use crate::{
    backends::common::{Allocation, Backend, Encoder, Kernels, kernel::TensorAddScaleKernel},
    config::{rope::AnyRoPEConfig, transformer::TransformerConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        mixer::{MixerState, attention::rope::PrecalculatedRoPE},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
        transformer_layer::{TransformerLayer, TransformerLayerError},
    },
    parameters::{ParameterLoaderError, ParameterTree},
    trace::{Array, RopeTap, TransformerLayerTap, TransformerTap, TransformerTapRequest},
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

pub struct TransformerEncodeOutput<B: Backend> {
    pub output: Option<Allocation<B>>,
    pub hidden_features: Option<Box<[Allocation<B>]>>,
    pub tap: TransformerTap<B>,
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
        encoder.push_debug_group("transformer accept");

        for layer_state in &mut self.layer_states {
            let TransformerLayerStateType::Owned(layer_state) = layer_state else {
                continue;
            };

            layer_state.encode_accept(accepted_indices, encoder)?;
        }

        self.context_length += accepted_indices.len();

        encoder.pop_debug_group();

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
    model_dim: usize,
    residual_add: <B::Kernels as Kernels>::TensorAddScaleKernel,
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
                    transformer_config.model_dim,
                    transformer_config.hidden_dim,
                    transformer_config.layer_configs.len(),
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
            ShortcutMode::Add,
            PostLayerScalar::None,
            data_type,
            &transformer_config.output_norm_config,
            &parameter_tree.subtree("output_norm")?,
            context,
        )?;

        let residual_add = <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, false)
            .map_err(TransformerNewError::Backend)?;

        Ok(Self {
            ropes: ropes.into_boxed_slice(),
            layers,
            output_norm,
            model_dim: transformer_config.model_dim,
            residual_add,
        })
    }

    fn data_type(&self) -> DataType {
        self.output_norm.data_type()
    }

    fn capture_residual(
        &self,
        shortcut: &Allocation<B>,
        hidden: &Allocation<B>,
        batch_size: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut output = encoder.allocate_scratch(hidden.size())?;
        let elements = (batch_size * self.model_dim) as u32;
        self.residual_add.encode(Some(shortcut), hidden, &mut output, elements, elements, 1.0, encoder);
        Ok(output)
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

    pub fn prefill_cache_layer_count(&self) -> usize {
        let num_layers = self.layers.len();
        let Some(last_owned_kv_layer_index) =
            self.layers.iter().rposition(|(layer, _rope_index)| layer.kv_source_layer_index.is_none())
        else {
            return num_layers;
        };

        last_owned_kv_layer_index + 1
    }

    pub fn prefill_cache_skips_trailing_layers(&self) -> bool {
        self.prefill_cache_layer_count() < self.layers.len()
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
        hidden_feature_layer_indices: Option<&[usize]>,
        mut state: Option<&mut TransformerState<B>>,
        tap_request: Option<&TransformerTapRequest>,
        encoder: &mut Encoder<B>,
    ) -> Result<TransformerEncodeOutput<B>, B::Error> {
        let request = tap_request.unwrap_or(&TransformerTapRequest::NONE);
        let layer_request = request.layers.as_ref();
        let mut tap = TransformerTap::default();

        let mut hidden = input;
        let layer_count = if output_range.is_none() && hidden_feature_layer_indices.is_none() {
            self.prefill_cache_layer_count()
        } else {
            self.layers.len()
        };

        let mut shortcut = encoder.allocate_scratch(hidden.size())?;
        let mut hidden_features =
            hidden_feature_layer_indices.map(|indices| (0..indices.len()).map(|_| None).collect::<Vec<_>>());

        let context_length = state.as_ref().map(|state| state.context_length).unwrap_or(0);
        let token_positions =
            batch_dim.heights().map(|rel_pos| context_length + rel_pos as usize).collect::<Box<[usize]>>();

        let precalculated_ropes = self
            .ropes
            .iter()
            .map(|rope_config| PrecalculatedRoPE::precalculate(rope_config, &token_positions, encoder))
            .collect::<Result<Box<[_]>, B::Error>>()?;

        if let Some(rope_request) = &request.rope_embeddings {
            for rope in precalculated_ropes.iter() {
                let shape = [1, token_positions.len(), rope.dim];
                tap.rope_embeddings.push(RopeTap {
                    cosines: rope_request
                        .cosines
                        .then(|| Array::capture(encoder, &rope.cosines, &shape, DataType::F32))
                        .transpose()?,
                    sines: rope_request
                        .sines
                        .then(|| Array::capture(encoder, &rope.sines, &shape, DataType::F32))
                        .transpose()?,
                });
            }
        }

        for (layer, layer_rope_index) in self.layers.iter().take(layer_count) {
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

            let layer_output = layer.encode(
                hidden,
                &mut shortcut,
                per_layer_inputs,
                precalculated_rope,
                batch_dim,
                layer_state,
                layer_request.and_then(|layer_request| layer_request.activations.as_ref()),
                encoder,
            )?;
            hidden = layer_output.hidden;

            if let Some(layer_request) = layer_request {
                // A layer's output is never materialized: the add is deferred into the
                // next layer's norm, so it has to be recomputed to be captured.
                let outputs = layer_request
                    .outputs
                    .then(|| {
                        let residual = self.capture_residual(&shortcut, &hidden, batch_dim.size(), encoder)?;
                        Array::capture(encoder, &residual, &[1, batch_dim.size(), self.model_dim], self.data_type())
                    })
                    .transpose()?;
                tap.layers.push(TransformerLayerTap {
                    outputs,
                    activations: Some(layer_output.tap),
                });
            }

            if let (Some(hidden_features), Some(indices)) = (&mut hidden_features, hidden_feature_layer_indices) {
                for (feature_index, &layer_index) in indices.iter().enumerate() {
                    if layer_index == layer.layer_index {
                        let feature = self.capture_residual(&shortcut, &hidden, batch_dim.size(), encoder)?;
                        hidden_features[feature_index] = Some(feature);
                    }
                }
            }
        }

        let hidden_features = hidden_features.map(|hidden_features| {
            hidden_features
                .into_iter()
                .enumerate()
                .map(|(feature_index, feature)| {
                    feature.unwrap_or_else(|| {
                        panic!(
                            "requested hidden feature for missing layer {}",
                            hidden_feature_layer_indices.unwrap()[feature_index]
                        )
                    })
                })
                .collect::<Box<[_]>>()
        });

        let Some(output_range) = output_range else {
            return Ok(TransformerEncodeOutput {
                output: None,
                hidden_features,
                tap,
            });
        };

        let output_normalized =
            self.output_norm.encode(&hidden, output_range.start, output_range.len(), Some(&mut shortcut), encoder)?;
        if request.output_norm {
            let shape = [1, output_range.len(), self.model_dim];
            tap.output_norm = Some(Array::capture(encoder, &output_normalized, &shape, self.data_type())?);
        }

        Ok(TransformerEncodeOutput {
            output: Some(output_normalized),
            hidden_features,
            tap,
        })
    }
}
