use std::{range::Range, sync::Arc};

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{Backend, Buffer, BufferRef, CommandBuffer, CommandBufferEncoding},
    config::{rope::AnyRoPEConfig, transformer::TransformerConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::InputRhtParameters,
        mixer::{MixerState, attention::mrope::MRoPEPositions},
        normalization::{Normalization, NormalizationNewError, ShortcutMode},
        residual::{ResidualEncodeError, ResidualHead, ResidualNewError},
        transformer_layer::{TransformerLayer, TransformerLayerError},
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
    rope_position: u32,
}

pub struct TransformerEncodeOutput<B: Backend> {
    pub output: Option<B::ScratchBuffer>,
    pub hidden_features: Option<Box<[B::ScratchBuffer]>>,
}

impl<B: Backend> TransformerState<B> {
    pub fn context_length(&self) -> u32 {
        self.context_length
    }

    pub fn rope_position(&self) -> u32 {
        self.rope_position
    }

    pub fn set_rope_position(
        &mut self,
        position: u32,
    ) {
        for layer_state in &mut self.layer_states {
            let TransformerLayerStateType::Owned(layer_state) = layer_state else {
                continue;
            };

            layer_state.set_rope_position(position);
        }

        self.rope_position = position;
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
            self.rope_position += accepted_indices.len() as u32;
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
    #[error("M-RoPE frequency pairs per axis sum to {frequency_pairs}, but the RoPE head has {head_pairs} pairs")]
    MRoPEFrequencyPairs {
        frequency_pairs: u32,
        head_pairs: u32,
    },
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
    ropes: Box<[AnyRoPEConfig]>,
    layers: Box<[TransformerLayer<B>]>,
    layer_ropes: Box<[Option<usize>]>,
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
        let mut ropes: Vec<AnyRoPEConfig> = Vec::new();

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
        let mut layer_ropes = Vec::with_capacity(num_layers as usize);
        for (layer_config, layer_index) in transformer_config.layer_configs.iter().zip(0u32..) {
            let layer_loader = parameter_tree.subtree(&format!("layers.{}", layer_index));

            if let Some(AnyRoPEConfig::MultimodalRoPEConfig(multimodal_config)) = &layer_config.rope_config {
                let frequency_pairs = multimodal_config.frequency_pairs_per_axis.iter().sum::<u32>();
                let head_pairs = multimodal_config.head_dim / 2;
                if frequency_pairs != head_pairs {
                    return Err(TransformerNewError::MRoPEFrequencyPairs {
                        frequency_pairs,
                        head_pairs,
                    });
                }
            }

            let rope = layer_config.rope_config.as_ref().map(|layer_rope_config| {
                ropes.iter().position(|rope_config| rope_config == layer_rope_config).unwrap_or_else(|| {
                    ropes.push(layer_rope_config.clone());
                    ropes.len() - 1
                })
            });

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
            layer_ropes.push(rope);
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
            ropes: ropes.into_boxed_slice(),
            layers: layers.into_boxed_slice(),
            layer_ropes: layer_ropes.into_boxed_slice(),
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
            rope_position: 0,
        })
    }

    pub fn encode(
        &self,
        input: B::ScratchBuffer,
        per_layer_inputs: Option<impl BufferRef<Backend = B>>,
        batch_dim: &BatchTopology<B>,
        output_range: Option<Range<u32>>,
        hidden_feature_layer_indices: Option<&[u32]>,
        state: Option<&mut TransformerState<B>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<TransformerEncodeOutput<B>, TransformerEncodeError<B>> {
        self.encode_at_positions(
            input,
            per_layer_inputs,
            None,
            batch_dim,
            output_range,
            hidden_feature_layer_indices,
            state,
            command_buffer,
        )
    }

    pub fn encode_at_positions(
        &self,
        input: B::ScratchBuffer,
        per_layer_inputs: Option<impl BufferRef<Backend = B>>,
        positions: Option<&MRoPEPositions>,
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

        let rope_positions = positions
            .map(|positions| {
                self.ropes
                    .iter()
                    .map(|rope_config| positions.precalculate(rope_config, command_buffer))
                    .collect::<Result<Box<[_]>, B::Error>>()
            })
            .transpose()
            .map_err(TransformerEncodeError::Backend)?;

        for layer in self.layers.iter().take(layer_count) {
            let layer_rope_positions = rope_positions
                .as_ref()
                .zip(self.layer_ropes[layer.layer_index as usize])
                .map(|(rope_positions, rope_index)| &rope_positions[rope_index]);

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

            stream = match layer_rope_positions {
                Some(positions) => layer.encode_at_positions(
                    stream,
                    per_layer_inputs,
                    positions,
                    batch_dim,
                    layer_state,
                    command_buffer,
                )?,
                None => layer.encode(stream, per_layer_inputs, batch_dim, layer_state, command_buffer)?,
            };

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
