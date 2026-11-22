use std::{range::Range, sync::Arc};

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{Backend, Buffer, BufferRef, CommandBuffer, CommandBufferEncoding},
    config::transformer::{LayerVisit, LoopConfig, TransformerConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::InputRhtParameters,
        mixer::MixerState,
        normalization::{Normalization, NormalizationNewError, ShortcutMode},
        residual::{ResidualEncodeError, ResidualHead, ResidualNewError},
        transformer_layer::{SublayerConvStates, TransformerLayer, TransformerLayerError},
    },
    safetensors::ParameterTree,
    utils::maybe_mut::MaybeMut,
};

enum TransformerLayerStateType<B: Backend> {
    Owned {
        mixer: Box<dyn MixerState<B>>,
        convs: Option<SublayerConvStates<B>>,
    },
    /// Shares the mixer state of the visit at this index.
    Shared(u32),
}

pub struct TransformerState<B: Backend> {
    layer_states: Box<[TransformerLayerStateType<B>]>,
    context_length: u32,
}

/// A hidden state of every row of a batch that a speculator reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HiddenFeature {
    /// The residual stream after a layer visit; with block attention residuals, its partial sum.
    AfterVisit(u32),
    /// The residual stream entering a layer visit, after a loop pass restarts it.
    BeforeVisit(u32),
    /// The output-normalized hidden state.
    Final,
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
            let TransformerLayerStateType::Owned {
                mixer,
                ..
            } = layer_state
            else {
                continue;
            };

            mixer.prepare(context_length, suffix_length)?;
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
                let TransformerLayerStateType::Owned {
                    mixer,
                    convs,
                } = layer_state
                else {
                    continue;
                };

                mixer.encode_accept(accepted_indices, command_buffer)?;
                if let Some(convs) = convs {
                    convs.encode_accept(accepted_indices, command_buffer);
                }
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
    #[error("Hidden features of a gated residual stream are not supported")]
    HiddenFeaturesUnsupportedResidual,
}

pub struct Transformer<B: Backend> {
    layers: Box<[TransformerLayer<B>]>,
    /// The layers in the order they run; a looped layer once per pass.
    visits: Box<[LayerVisit]>,
    loop_config: Option<LoopConfig>,
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
        let num_layers = transformer_config.layer_configs.len() as u32;
        let loop_config = transformer_config.loop_config.clone();
        let residual_head = ResidualHead::new(
            &residual_config,
            transformer_config.model_dim,
            num_layers,
            loop_config.as_ref(),
            input_data_type,
            &parameter_tree.subtree("residual_head"),
            context,
        )
        .await?;

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
                loop_config.as_ref(),
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
            visits: transformer_config.layer_visits(),
            loop_config,
            output_norm,
            model_dim: transformer_config.model_dim,
            residual_head,
        })
    }

    fn capture_hidden_feature(
        &self,
        stream: &B::ScratchBuffer,
        token_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, TransformerEncodeError<B>> {
        let rows = self
            .residual_head
            .hidden_rows(stream.size(), token_count)
            .ok_or(TransformerEncodeError::HiddenFeaturesUnsupportedResidual)?;
        let mut copy =
            command_buffer.allocate_scratch(rows.end - rows.start).map_err(TransformerEncodeError::Backend)?;
        command_buffer.encode_copy(stream.subrange(rows), &mut copy);
        Ok(copy)
    }

    /// The output-normalized hidden state of rows `rows` of `stream`, without consuming it.
    fn encode_output_rows(
        &self,
        stream: &B::ScratchBuffer,
        token_count: u32,
        rows: Range<u32>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, TransformerEncodeError<B>> {
        let row_count = rows.end - rows.start;
        let readout = self.residual_head.readout(stream, token_count, rows.start, row_count, command_buffer)?;
        match (&self.output_norm, readout) {
            (Some(output_norm), readout) => {
                let (hidden, row_offset) = match &readout {
                    Some(hidden) => (hidden, 0),
                    None => (stream, rows.start),
                };
                output_norm
                    .encode(hidden, row_offset, row_count, None::<&mut B::ScratchBuffer>, command_buffer)
                    .map_err(TransformerEncodeError::Backend)
            },
            (None, Some(hidden)) => Ok(hidden),
            (None, None) => {
                let row_size = stream.size() / token_count as usize;
                let bytes = rows.start as usize * row_size..rows.end as usize * row_size;
                let mut output = command_buffer
                    .allocate_scratch(bytes.end - bytes.start)
                    .map_err(TransformerEncodeError::Backend)?;
                command_buffer.encode_copy(stream.subrange(bytes), &mut output);
                Ok(output)
            },
        }
    }

    pub fn model_dim(&self) -> u32 {
        self.model_dim
    }

    pub fn layer_visit_count(&self) -> u32 {
        self.visits.len() as u32
    }

    pub fn speculation_supported(&self) -> bool {
        !matches!(self.residual_head, ResidualHead::Gated(_))
            && self.layers.iter().all(|layer| layer.mixer.speculation_supported())
    }

    pub fn max_context_length(&self) -> Option<u32> {
        self.layers.iter().map(|layer| layer.mixer.max_context_length()).fold(None, |acc, el| match (acc, el) {
            (Some(a), Some(b)) => Some(u32::min(a, b)),
            (Some(x), None) | (None, Some(x)) => Some(x),
            (None, None) => None,
        })
    }

    /// How many layer visits a prefill without outputs has to run to fill every cache.
    pub fn prefill_cache_layer_count(&self) -> usize {
        if self.loop_config.is_some() {
            return self.visits.len();
        }
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
            .visits
            .iter()
            .map(|visit| {
                let layer = &self.layers[visit.layer_index as usize];
                match layer.kv_source_layer_index {
                    None => Ok(TransformerLayerStateType::Owned {
                        mixer: layer.mixer.create_empty_state(max_context_length, context)?,
                        convs: layer.create_conv_states(context)?,
                    }),
                    Some(kv_source_layer_index) => Ok(TransformerLayerStateType::Shared(kv_source_layer_index)),
                }
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
        requested_features: Option<&[HiddenFeature]>,
        mut state: Option<&mut TransformerState<B>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<TransformerEncodeOutput<B>, TransformerEncodeError<B>> {
        let token_count = batch_dim.size();
        let mut stream = self
            .residual_head
            .expand(input, token_count, command_buffer)
            .map_err(TransformerEncodeError::Backend)?;
        let visit_count = if output_range.is_none() && requested_features.is_none() {
            self.prefill_cache_layer_count()
        } else {
            self.visits.len()
        };

        let mut hidden_features =
            requested_features.map(|features| (0..features.len()).map(|_| None).collect::<Vec<_>>());
        let capture = |feature: HiddenFeature,
                           stream: &B::ScratchBuffer,
                           hidden_features: &mut Option<Vec<Option<B::ScratchBuffer>>>,
                           command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding|
         -> Result<(), TransformerEncodeError<B>> {
            let (Some(hidden_features), Some(requested)) = (hidden_features, requested_features) else {
                return Ok(());
            };
            for (slot, _) in hidden_features.iter_mut().zip(requested).filter(|(_, requested)| **requested == feature) {
                *slot = Some(match feature {
                    HiddenFeature::Final => {
                        self.encode_output_rows(stream, token_count, (0..token_count).into(), command_buffer)?
                    },
                    HiddenFeature::AfterVisit(_) | HiddenFeature::BeforeVisit(_) => {
                        self.capture_hidden_feature(stream, token_count, command_buffer)?
                    },
                });
            }
            Ok(())
        };
        let mut loop_input = None;

        for (visit_index, visit) in self.visits.iter().enumerate().take(visit_count) {
            let layer = &self.layers[visit.layer_index as usize];
            if let Some(loop_config) = &self.loop_config
                && visit.layer_index == loop_config.start_layer
            {
                let ResidualHead::BlockAttention(head) = &self.residual_head else {
                    panic!("looped layers need block attention residuals");
                };
                if visit.loop_pass == 0 {
                    let mut saved =
                        command_buffer.allocate_scratch(stream.size()).map_err(TransformerEncodeError::Backend)?;
                    command_buffer.encode_copy(&stream, &mut saved);
                    loop_input = Some(saved);
                } else {
                    let loop_input = loop_input.as_ref().expect("a repeated loop pass before the first one");
                    let num_loop_input_blocks = loop_config.start_layer.div_ceil(head.layout().block_size);
                    stream = head
                        .begin_loop_pass(loop_input, &stream, num_loop_input_blocks, token_count, command_buffer)
                        .map_err(TransformerEncodeError::Backend)?;
                }
            }

            let (layer_state, conv_states) = if let Some(state) = &mut state {
                let (owned, shared) = state.layer_states.split_at_mut(visit_index);
                match &mut shared[0] {
                    TransformerLayerStateType::Owned {
                        mixer,
                        convs,
                    } => (Some(MaybeMut::Mut(mixer.as_mut())), convs.as_mut()),
                    TransformerLayerStateType::Shared(owned_layer_index) => {
                        let TransformerLayerStateType::Owned {
                            mixer,
                            ..
                        } = &owned[*owned_layer_index as usize]
                        else {
                            panic!("shared layer doesn't point to an owned layer");
                        };
                        (Some(MaybeMut::Const(mixer.as_ref())), None)
                    },
                }
            } else {
                (None, None)
            };

            capture(HiddenFeature::BeforeVisit(visit_index as u32), &stream, &mut hidden_features, command_buffer)?;
            stream = layer.encode(
                stream,
                per_layer_inputs,
                batch_dim,
                layer_state,
                conv_states,
                visit.loop_pass > 0,
                command_buffer,
            )?;
            capture(HiddenFeature::AfterVisit(visit_index as u32), &stream, &mut hidden_features, command_buffer)?;
        }
        capture(HiddenFeature::Final, &stream, &mut hidden_features, command_buffer)?;

        let hidden_features = hidden_features.map(|hidden_features| {
            hidden_features
                .into_iter()
                .zip(requested_features.unwrap())
                .map(|(feature, requested)| {
                    feature.unwrap_or_else(|| panic!("requested hidden feature {requested:?} was not produced"))
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
            match self.residual_head.readout(
                &stream,
                token_count,
                output_range.start,
                output_row_count,
                command_buffer,
            )? {
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
