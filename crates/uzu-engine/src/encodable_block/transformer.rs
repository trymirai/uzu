use std::{range::Range, sync::Arc};

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{Backend, Buffer, BufferRef, CommandBuffer, CommandBufferEncoding},
    config::transformer::TransformerConfig,
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::InputRhtParameters,
        mixer::MixerState,
        normalization::{Normalization, NormalizationNewError, ShortcutMode},
        residual::{ResidualEncodeError, ResidualHead, ResidualNewError},
        transformer_layer::{StreamOpStates, TransformerLayer, TransformerLayerError},
    },
    safetensors::ParameterTree,
    utils::maybe_mut::MaybeMut,
};

enum LayerMixerState<B: Backend> {
    Owned(Box<dyn MixerState<B>>),
    Shared(u32),
}

struct TransformerLayerState<B: Backend> {
    mixer: LayerMixerState<B>,
    stream_ops: StreamOpStates<B>,
}

pub struct TransformerState<B: Backend> {
    layer_states: Box<[TransformerLayerState<B>]>,
    context_length: u32,
}

pub struct TransformerEncodeOutput<B: Backend> {
    pub output: Option<B::ScratchBuffer>,
    pub hidden_features: Option<Box<[B::ScratchBuffer]>>,
}

impl<B: Backend> TransformerState<B> {
    pub fn context_length(&self) -> u32 {
        self.context_length
    }

    pub fn prepare(
        &mut self,
        context_length: u32,
        suffix_length: u32,
    ) -> Result<(), B::Error> {
        for layer_state in &mut self.layer_states {
            let LayerMixerState::Owned(mixer_state) = &mut layer_state.mixer else {
                continue;
            };

            mixer_state.prepare(context_length, suffix_length)?;
        }

        Ok(())
    }

    pub fn encode_accept(
        &mut self,
        accepted_indices: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        {
            let command_buffer = &mut *command_buffer.debug_group("transformer accept");

            for layer_state in &mut self.layer_states {
                if let LayerMixerState::Owned(mixer_state) = &mut layer_state.mixer {
                    mixer_state.encode_accept(accepted_indices, command_buffer)?;
                }

                layer_state.stream_ops.encode_accept(accepted_indices, command_buffer)?;
            }

            self.context_length += accepted_indices.len() as u32;
        }

        command_buffer.debug_point(&format!("accepted {} tokens", accepted_indices.len()));

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum TransformerNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Layer error: {0}")]
    Layer(#[from] TransformerLayerError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("Residual error: {0}")]
    Residual(#[from] ResidualNewError<B>),
    #[error("Readout expects RHT-rotated input but there is no output norm to apply it")]
    UnappliedReadoutRht,
}

#[derive(Debug, Error)]
pub enum TransformerEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Layer error: {0}")]
    Layer(#[from] TransformerLayerError<B>),
    #[error("Residual encoding error: {0}")]
    Residual(#[from] ResidualEncodeError<B>),
    #[error("Hidden features require an additive residual")]
    HiddenFeaturesRequireAdditiveResidual,
}

pub struct Transformer<B: Backend> {
    layers: Box<[TransformerLayer<B>]>,
    output_norm: Option<Normalization<B>>,
    model_dim: u32,
    residual_head: ResidualHead<B>,
}

impl<B: Backend> Transformer<B> {
    pub async fn new(
        input_data_type: DataType,
        output_data_type: DataType,
        output_norm_hadamard_factors: Option<Arc<B::GlobalGpuBuffer>>,
        transformer_config: &TransformerConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, TransformerNewError<B>> {
        let residual_config = transformer_config.residual_config.clone();
        let residual_head = ResidualHead::new(
            &residual_config,
            transformer_config.model_dim,
            input_data_type,
            &parameter_tree.subtree("residual_head"),
            context,
        )
        .await?;

        let num_layers = transformer_config.layer_configs.len() as u32;
        let mut layers = Vec::with_capacity(num_layers as usize);
        for (layer_config, layer_index) in transformer_config.layer_configs.iter().zip(0u32..) {
            let layer_loader = parameter_tree.subtree(&format!("layers.{}", layer_index));

            let layer = TransformerLayer::new(
                transformer_config.model_dim,
                transformer_config.hidden_dim,
                num_layers,
                layer_index,
                input_data_type,
                layer_config,
                &residual_config,
                &layer_loader,
                context,
            )
            .await?;

            layers.push(layer);
        }

        let output_norm = match (&transformer_config.output_norm_config, output_norm_hadamard_factors) {
            (Some(output_norm_config), hadamard_factors) => Some(
                Normalization::new(
                    transformer_config.model_dim,
                    input_data_type,
                    output_data_type,
                    hadamard_factors.map(InputRhtParameters::h32),
                    ShortcutMode::None,
                    output_norm_config,
                    &parameter_tree.subtree("output_norm"),
                    context,
                )
                .await?,
            ),
            (None, Some(_)) => return Err(TransformerNewError::UnappliedReadoutRht),
            (None, None) => None,
        };

        Ok(Self {
            layers: layers.into_boxed_slice(),
            output_norm,
            model_dim: transformer_config.model_dim,
            residual_head,
        })
    }

    fn capture_hidden_feature(
        &self,
        stream: impl BufferRef<Backend = B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut copy = command_buffer.allocate_scratch(stream.size())?;
        command_buffer.encode_copy(stream, &mut copy);
        Ok(copy)
    }

    pub fn model_dim(&self) -> u32 {
        self.model_dim
    }

    pub fn speculation_supported(&self) -> bool {
        matches!(self.residual_head, ResidualHead::Additive)
            && self.layers.iter().all(|layer| layer.speculation_supported())
    }

    pub fn max_context_length(&self) -> Option<u32> {
        self.layers.iter().map(|layer| layer.mixer.max_context_length()).fold(None, |acc, el| match (acc, el) {
            (Some(a), Some(b)) => Some(u32::min(a, b)),
            (Some(x), None) | (None, Some(x)) => Some(x),
            (None, None) => None,
        })
    }

    pub fn prefill_cache_layer_count(&self) -> usize {
        let num_layers = self.layers.len();
        let Some(last_stateful_layer_index) = self
            .layers
            .iter()
            .rposition(|layer| layer.kv_source_layer_index.is_none() || layer.has_ngram_embeddings())
        else {
            return num_layers;
        };

        last_stateful_layer_index + 1
    }

    pub fn prefill_cache_skips_trailing_layers(&self) -> bool {
        self.prefill_cache_layer_count() < self.layers.len()
    }

    pub fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<TransformerState<B>, B::Error> {
        let layer_states = self
            .layers
            .iter()
            .map(|layer| {
                let mixer = match layer.kv_source_layer_index {
                    None => LayerMixerState::Owned(layer.mixer.create_empty_state(max_context_length, context)?),
                    Some(kv_source_layer_index) => LayerMixerState::Shared(kv_source_layer_index),
                };
                Ok(TransformerLayerState {
                    mixer,
                    stream_ops: layer.create_stream_op_states(context)?,
                })
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
        input: B::ScratchBuffer,
        per_layer_inputs: Option<impl BufferRef<Backend = B>>,
        batch_dim: &BatchTopology<B>,
        output_range: Option<Range<u32>>,
        hidden_feature_layer_indices: Option<&[u32]>,
        mut state: Option<&mut TransformerState<B>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<TransformerEncodeOutput<B>, TransformerEncodeError<B>> {
        if hidden_feature_layer_indices.is_some() && !matches!(self.residual_head, ResidualHead::Additive) {
            return Err(TransformerEncodeError::HiddenFeaturesRequireAdditiveResidual);
        }

        let mut stream = self
            .residual_head
            .expand(input, batch_dim.size(), command_buffer)
            .map_err(TransformerEncodeError::Backend)?;
        let layer_count = if output_range.is_none() && hidden_feature_layer_indices.is_none() {
            self.prefill_cache_layer_count()
        } else {
            self.layers.len()
        };

        let mut hidden_features =
            hidden_feature_layer_indices.map(|indices| (0..indices.len()).map(|_| None).collect::<Vec<_>>());

        for layer in self.layers.iter().take(layer_count) {
            let layer_state = state.as_deref_mut().map(|state| {
                let layer_index = layer.layer_index as usize;
                if let LayerMixerState::Shared(owner_index) = state.layer_states[layer_index].mixer {
                    let [layer_state, owner_state] =
                        state.layer_states.get_disjoint_mut([layer_index, owner_index as usize]).unwrap();
                    let LayerMixerState::Owned(owner_mixer_state) = &owner_state.mixer else {
                        panic!("shared layer doesn't point to an owned layer");
                    };
                    (MaybeMut::Const(owner_mixer_state.as_ref()), Some(&mut layer_state.stream_ops))
                } else {
                    let TransformerLayerState {
                        mixer: LayerMixerState::Owned(mixer_state),
                        stream_ops,
                    } = &mut state.layer_states[layer_index]
                    else {
                        unreachable!()
                    };
                    (MaybeMut::Mut(mixer_state.as_mut()), Some(stream_ops))
                }
            });

            stream = layer.encode(stream, per_layer_inputs, batch_dim, layer_state, command_buffer)?;

            if let (Some(hidden_features), Some(indices)) = (&mut hidden_features, hidden_feature_layer_indices) {
                for (feature_index, &layer_index) in indices.iter().enumerate() {
                    if layer_index == layer.layer_index {
                        let feature = self
                            .capture_hidden_feature(&stream, command_buffer)
                            .map_err(TransformerEncodeError::Backend)?;
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
            });
        };

        let output_row_count = output_range.end - output_range.start;
        let (hidden, row_offset) =
            match self.residual_head.readout(&stream, output_range.start, output_row_count, command_buffer)? {
                Some(hidden) => (hidden, 0),
                None => (stream, output_range.start),
            };
        let output = match &self.output_norm {
            Some(output_norm) => output_norm
                .encode(&hidden, row_offset, output_row_count, None::<&mut B::ScratchBuffer>, command_buffer)
                .map_err(TransformerEncodeError::Backend)?,
            None if row_offset == 0 => hidden,
            None => {
                let row_size = hidden.size() / batch_dim.size() as usize;
                let rows = row_offset as usize * row_size..(row_offset + output_row_count) as usize * row_size;
                let mut output =
                    command_buffer.allocate_scratch(rows.end - rows.start).map_err(TransformerEncodeError::Backend)?;
                command_buffer.encode_copy((&hidden).subrange(rows), &mut output);
                output
            },
        };

        Ok(TransformerEncodeOutput {
            output: Some(output),
            hidden_features,
        })
    }

    pub fn has_ngram_embeddings(&self) -> bool {
        self.layers.iter().any(|layer| layer.has_ngram_embeddings())
    }

    pub fn prepare_ngram_rows(
        &self,
        state: &mut TransformerState<B>,
        token_ids: &[u32],
        parents: Option<&[i32]>,
    ) -> std::io::Result<()> {
        for (op, op_state) in self
            .layers
            .iter()
            .zip(state.layer_states.iter_mut())
            .flat_map(|(layer, layer_state)| layer.ngram_injections(&mut layer_state.stream_ops))
        {
            op.prepare(op_state, token_ids, parents)?;
        }
        Ok(())
    }
}
