use std::{range::Range, sync::Arc};

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{Backend, Buffer, BufferRef, CommandBuffer, CommandBufferEncoding},
    config::{
        token_mixer::AnyTokenMixerConfig, transformer::TransformerConfig, transformer_layer::TransformerLayerConfig,
    },
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::InputRhtParameters,
        mixer::MixerState,
        normalization::{Normalization, NormalizationNewError, ShortcutMode},
        residual::{ResidualEncodeError, ResidualHead, ResidualNewError},
        transformer_layer::{PendingMlp, TransformerLayer, TransformerLayerError},
    },
    safetensors::ParameterTree,
    utils::maybe_mut::MaybeMut,
};

enum TransformerLayerStateType<B: Backend> {
    Owned(Box<dyn MixerState<B>>),
    Shared(u32),
}

pub struct TransformerState<B: Backend> {
    layer_states: Box<[TransformerLayerStateType<B>]>,
    context_length: u32,
}

enum ChunkState<'m, B: Backend> {
    Stream(B::ScratchBuffer),
    MlpPending(PendingMlp<'m, B>),
}

struct Chunk<'m, B: Backend> {
    state: Option<ChunkState<'m, B>>,
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
            let TransformerLayerStateType::Owned(layer_state) = layer_state else {
                continue;
            };

            layer_state.prepare(context_length, suffix_length)?;
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
                let TransformerLayerStateType::Owned(layer_state) = layer_state else {
                    continue;
                };

                layer_state.encode_accept(accepted_indices, command_buffer)?;
            }

            self.context_length += accepted_indices.len() as u32;
        }

        command_buffer.debug_point(&format!("accepted {} tokens", accepted_indices.len()));

        Ok(())
    }

    fn encode_accept_layer(
        &mut self,
        layer_index: u32,
        accepted_indices: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        if let TransformerLayerStateType::Owned(layer_state) = &mut self.layer_states[layer_index as usize] {
            layer_state.encode_accept(accepted_indices, command_buffer)?;
        }
        Ok(())
    }

    fn layer_state(
        &mut self,
        layer_index: u32,
    ) -> MaybeMut<'_, dyn MixerState<B>> {
        let layer_index = layer_index as usize;
        let owned_layer_index = match &self.layer_states[layer_index] {
            TransformerLayerStateType::Owned(_) => None,
            TransformerLayerStateType::Shared(owned_layer_index) => Some(*owned_layer_index as usize),
        };

        match owned_layer_index {
            Some(owned_layer_index) => {
                let TransformerLayerStateType::Owned(owned_layer) = &self.layer_states[owned_layer_index] else {
                    panic!("shared layer doesn't point to an owned layer");
                };
                MaybeMut::Const(owned_layer.as_ref())
            },
            None => {
                let TransformerLayerStateType::Owned(layer_state) = &mut self.layer_states[layer_index] else {
                    unreachable!("layer state is still owned");
                };
                MaybeMut::Mut(layer_state.as_mut())
            },
        }
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
    #[error("prefill chunk groups support at most two chunks (got {0})")]
    TooManyPrefillChunks(usize),
}

pub struct Transformer<B: Backend> {
    layers: Box<[TransformerLayer<B>]>,
    output_norm: Option<Normalization<B>>,
    model_dim: u32,
    residual_head: ResidualHead<B>,
    supports_layerwise_prefill_pipelining: bool,
}

fn layer_pipelines_prefill(layer_config: &TransformerLayerConfig) -> bool {
    let mixer_pipelines = match &layer_config.mixer.body_config {
        AnyTokenMixerConfig::DeltaNetConfig(_) => true,
        AnyTokenMixerConfig::AttentionConfig(attention) => {
            attention.sliding_window_size.is_none() && !attention.is_kv_sharing
        },
        AnyTokenMixerConfig::ShortConvConfig(_) | AnyTokenMixerConfig::SparseAttentionConfig(_) => false,
    };
    mixer_pipelines
        && layer_config.kv_source_layer_index.is_none()
        && layer_config.mixer.common.conv_config.is_none()
        && layer_config.mlp.common.conv_config.is_none()
        && layer_config.prologue_configs.is_empty()
        && layer_config.epilogue_configs.is_empty()
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
            supports_layerwise_prefill_pipelining: transformer_config.layer_configs.iter().all(layer_pipelines_prefill),
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
            && self.layers.iter().all(|layer| layer.mixer.speculation_supported())
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
        let Some(last_owned_kv_layer_index) =
            self.layers.iter().rposition(|layer| layer.kv_source_layer_index.is_none())
        else {
            return num_layers;
        };

        last_owned_kv_layer_index + 1
    }

    pub fn prefill_cache_skips_trailing_layers(&self) -> bool {
        self.prefill_cache_layer_count() < self.layers.len()
    }

    pub fn supports_layerwise_prefill_pipelining(&self) -> bool {
        self.supports_layerwise_prefill_pipelining
    }

    pub fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<TransformerState<B>, B::Error> {
        let layer_states = self
            .layers
            .iter()
            .map(|layer| match layer.kv_source_layer_index {
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
        input: B::ScratchBuffer,
        per_layer_inputs: Option<&B::ScratchBuffer>,
        batch_dim: &BatchTopology<B>,
        output_range: Option<Range<u32>>,
        hidden_feature_layer_indices: Option<&[u32]>,
        state: Option<&mut TransformerState<B>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<TransformerEncodeOutput<B>, TransformerEncodeError<B>> {
        self.encode_chunks(
            vec![input],
            per_layer_inputs,
            &[batch_dim],
            output_range,
            hidden_feature_layer_indices,
            state,
            command_buffer,
        )
    }

    pub fn encode_chunks(
        &self,
        inputs: Vec<B::ScratchBuffer>,
        per_layer_inputs: Option<&B::ScratchBuffer>,
        batch_dims: &[&BatchTopology<B>],
        output_range: Option<Range<u32>>,
        hidden_feature_layer_indices: Option<&[u32]>,
        state: Option<&mut TransformerState<B>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<TransformerEncodeOutput<B>, TransformerEncodeError<B>> {
        assert!(!inputs.is_empty(), "prefill needs at least one chunk");
        assert_eq!(inputs.len(), batch_dims.len(), "each prefill chunk needs a batch topology");
        if inputs.len() > 2 {
            return Err(TransformerEncodeError::TooManyPrefillChunks(inputs.len()));
        }
        if inputs.len() == 1 {
            let input = inputs.into_iter().next().unwrap();
            return self.encode_single(
                input,
                per_layer_inputs,
                batch_dims[0],
                output_range,
                hidden_feature_layer_indices,
                state,
                command_buffer,
            );
        }
        assert!(self.supports_layerwise_prefill_pipelining, "this model's chunks cannot be overlapped");
        assert!(state.is_some(), "overlapped chunks require mixer state");

        if hidden_feature_layer_indices.is_some() && !matches!(self.residual_head, ResidualHead::Additive) {
            return Err(TransformerEncodeError::HiddenFeaturesRequireAdditiveResidual);
        }

        let mut state = state;
        let initial_context_length = state.as_ref().map_or(0, |state| state.context_length);
        let mut context_length = initial_context_length;
        let mut chunks = Vec::with_capacity(inputs.len());
        for (input, batch_dim) in inputs.into_iter().zip(batch_dims) {
            let stream = self
                .residual_head
                .expand(input, batch_dim.size(), command_buffer)
                .map_err(TransformerEncodeError::Backend)?;
            context_length += batch_dim.size();
            chunks.push(Chunk {
                state: Some(ChunkState::Stream(stream)),
            });
        }
        let last = chunks.len() - 1;
        let layer_count = if output_range.is_none() && hidden_feature_layer_indices.is_none() {
            self.prefill_cache_layer_count()
        } else {
            self.layers.len()
        };
        let mut hidden_features =
            hidden_feature_layer_indices.map(|indices| (0..indices.len()).map(|_| None).collect::<Vec<_>>());

        let result = (|| {
            for step in 0..=layer_count {
                for index in 0..chunks.len() {
                    if step > 0 {
                        let layer = &self.layers[step - 1];
                        let Some(ChunkState::MlpPending(pending)) = chunks[index].state.take() else {
                            unreachable!("layer MLP hand-off is pending")
                        };
                        let stream =
                            layer.encode_from_mlp(pending, per_layer_inputs, batch_dims[index], command_buffer)?;
                        chunks[index].state = Some(ChunkState::Stream(stream));

                        if index == last
                            && let (Some(hidden_features), Some(indices)) =
                                (&mut hidden_features, hidden_feature_layer_indices)
                        {
                            for (feature_index, &layer_index) in indices.iter().enumerate() {
                                if layer_index == layer.layer_index {
                                    let feature = self
                                        .capture_hidden_feature(
                                            match chunks[index].state.as_ref() {
                                                Some(ChunkState::Stream(stream)) => stream,
                                                _ => unreachable!("completed layer stream is available"),
                                            },
                                            command_buffer,
                                        )
                                        .map_err(TransformerEncodeError::Backend)?;
                                    hidden_features[feature_index] = Some(feature);
                                }
                            }
                        }
                    }

                    if step < layer_count {
                        let layer = &self.layers[step];
                        let layer_state = state.as_deref_mut().map(|state| state.layer_state(layer.layer_index));
                        let Some(ChunkState::Stream(stream)) = chunks[index].state.take() else {
                            unreachable!("previous layer stream is available")
                        };
                        let stream = layer.encode_before_mlp(
                            stream,
                            per_layer_inputs,
                            batch_dims[index],
                            layer_state,
                            command_buffer,
                        )?;
                        let pending = layer.encode_until_mlp(stream, batch_dims[index].size(), command_buffer)?;
                        chunks[index].state = Some(ChunkState::MlpPending(pending));

                        if index < last {
                            let accepted = (0..batch_dims[index].size()).collect::<Box<[u32]>>();
                            state
                                .as_deref_mut()
                                .expect("overlapped chunks require state")
                                .encode_accept_layer(layer.layer_index, &accepted, command_buffer)
                                .map_err(TransformerEncodeError::Backend)?;
                        }
                    }
                }
            }
            Ok::<(), TransformerEncodeError<B>>(())
        })();
        if let Err(error) = result {
            for chunk in &mut chunks {
                if let Some(ChunkState::MlpPending(pending)) = chunk.state.take() {
                    pending.abandon(command_buffer);
                }
            }
            return Err(error);
        }

        if let Some(state) = state {
            state.context_length = context_length - batch_dims[last].size();
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

        let Some(ChunkState::Stream(last_stream)) = chunks[last].state.take() else {
            unreachable!("last chunk completed all layers")
        };
        let output_row_count = output_range.end - output_range.start;
        let (hidden, row_offset) =
            match self.residual_head.readout(&last_stream, output_range.start, output_row_count, command_buffer)? {
                Some(hidden) => (hidden, 0),
                None => (last_stream, output_range.start),
            };
        let output = match &self.output_norm {
            Some(output_norm) => output_norm
                .encode(&hidden, row_offset, output_row_count, None::<&mut B::ScratchBuffer>, command_buffer)
                .map_err(TransformerEncodeError::Backend)?,
            None if row_offset == 0 => hidden,
            None => {
                let row_size = hidden.size() / batch_dims[last].size() as usize;
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

    fn encode_single(
        &self,
        input: B::ScratchBuffer,
        per_layer_inputs: Option<&B::ScratchBuffer>,
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
            let layer_state = if let Some(state) = &mut state {
                Some(match &mut state.layer_states[layer.layer_index as usize] {
                    TransformerLayerStateType::Owned(layer_state) => MaybeMut::Mut(layer_state.as_mut()),
                    TransformerLayerStateType::Shared(owned_layer_index) => {
                        let TransformerLayerStateType::Owned(owned_layer) =
                            &state.layer_states[*owned_layer_index as usize]
                        else {
                            panic!("shared layer doesn't point to an owned layer");
                        };
                        MaybeMut::Const(owned_layer.as_ref())
                    },
                })
            } else {
                None
            };

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
}
