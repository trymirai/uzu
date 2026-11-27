use std::{cmp::Reverse, range::Range, sync::Arc};

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
        transformer_layer::{TransformerLayer, TransformerLayerError},
    },
    safetensors::ParameterTree,
    utils::maybe_mut::MaybeMut,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct LayerId(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StateId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CacheStep {
    layer: LayerId,
    state: StepState,
    loop_action: Option<LoopAction>,
    captures_hidden: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoopAction {
    /// A step will restart into this step.
    Start,
    Restart,
}

/// A step that does *not* write anything into cache
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ReadoutStep {
    layer: LayerId,
    state: StateId,
    captures_hidden: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepState {
    Owned(StateId),
    Shared(StateId),
}

#[derive(Debug, Clone)]
struct LayerSchedule {
    cache_steps: Box<[CacheStep]>,
    readout_steps: Box<[ReadoutStep]>,
    owned_cache_count: u32,
    hidden_capture_count: u32,
    /// whether any of the readout steps have hidden layers that need to be captured
    has_readout_hiddens: bool,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ScheduleError {
    #[error("Transformer has no steps to run")]
    NoSteps,
    #[error("Hidden layer capture {0} is out of order")]
    HiddenIndicesOutOfOrder(u32),
    #[error("Layer {0}'s state points to layer {1}, which does not own state")]
    SharedNotIntoOwned(u32, u32),
    #[error("Layer {0} has a shared KV cache, which is incompatible with looping architectures")]
    LoopingAndSharedKVCache(u32),
    #[error("Tried to reference layer index {index} out of {len} layers")]
    LayerIndexOutOfBounds {
        index: u32,
        len: usize,
    },
    #[error("Looped layer indices skip layer {0}")]
    SkippedLayerIndex(u32),
}

impl LayerSchedule {
    /// `kv_sources[layer]` is that layer's `kv_source_layer_index`.
    fn new(
        kv_sources: &[Option<u32>],
        layer_order: Option<&[u32]>,
        hidden_feature_layer_indices: Option<&[u32]>,
    ) -> Result<Self, ScheduleError> {
        if layer_order.map_or(kv_sources.is_empty(), <[u32]>::is_empty) {
            return Err(ScheduleError::NoSteps);
        }

        let mut cache_steps = Vec::with_capacity(layer_order.map_or(kv_sources.len(), <[u32]>::len));
        let mut hidden_capture_count = 0;

        let mut hidden_targets = hidden_feature_layer_indices.into_iter().flatten();
        let mut hidden_target = hidden_targets.next();
        let mut captures_hidden = |step_index: usize| {
            if hidden_target != Some(&(step_index as u32)) {
                return false;
            }

            hidden_capture_count += 1;
            hidden_target = hidden_targets.next();

            true
        };

        let owned_cache_count = if let Some(indices) = layer_order {
            for (i, &layer_index) in indices.iter().enumerate() {
                let kv_source = kv_sources.get(layer_index as usize).ok_or(ScheduleError::LayerIndexOutOfBounds {
                    index: layer_index,
                    len: kv_sources.len(),
                })?;

                if kv_source.is_some() {
                    return Err(ScheduleError::LoopingAndSharedKVCache(layer_index));
                }

                let step = CacheStep {
                    layer: LayerId(layer_index),
                    state: StepState::Owned(StateId(i as u32)),
                    loop_action: None,
                    captures_hidden: captures_hidden(i),
                };

                cache_steps.push(step);
            }

            indices.len() as u32
        } else {
            let mut state_index = 0;
            let mut layer_to_state_id = vec![None; kv_sources.len()];
            for (layer_index, &kv_source) in kv_sources.iter().enumerate() {
                let step = CacheStep {
                    layer: LayerId(layer_index as u32),
                    state: match kv_source {
                        Some(source_layer_index) => StepState::Shared(StateId(
                            layer_to_state_id
                                .get(source_layer_index as usize)
                                .ok_or(ScheduleError::LayerIndexOutOfBounds {
                                    index: source_layer_index,
                                    len: kv_sources.len(),
                                })?
                                .ok_or(ScheduleError::SharedNotIntoOwned(layer_index as u32, source_layer_index))?,
                        )),
                        None => {
                            let id = StateId(state_index);
                            debug_assert!(layer_to_state_id[layer_index].is_none());
                            layer_to_state_id[layer_index] = Some(state_index);
                            state_index += 1;
                            StepState::Owned(id)
                        },
                    },
                    loop_action: None,
                    captures_hidden: captures_hidden(layer_index),
                };

                cache_steps.push(step);
            }

            state_index
        };

        if let Some(target) = hidden_target {
            return Err(ScheduleError::HiddenIndicesOutOfOrder(*target));
        }

        // No need to check for looping and other invariants unless we have custom order
        if layer_order.is_some() {
            let mut max_so_far = cache_steps[0].layer;
            if max_so_far != LayerId(0) {
                return Err(ScheduleError::SkippedLayerIndex(0));
            }
            let mut starts = Vec::new();
            for step in 1..cache_steps.len() {
                let ([.., prev], [next, ..]) = cache_steps.split_at_mut(step) else {
                    unreachable!()
                };

                if next.layer > max_so_far {
                    if next.layer.0 != max_so_far.0 + 1 {
                        return Err(ScheduleError::SkippedLayerIndex(max_so_far.0 + 1));
                    }
                    max_so_far = next.layer;
                }

                if next.layer <= prev.layer {
                    next.loop_action = Some(LoopAction::Restart);
                    starts.push(next.layer);
                }
            }

            if max_so_far.0 != kv_sources.len() as u32 - 1 {
                return Err(ScheduleError::SkippedLayerIndex(max_so_far.0 + 1));
            }

            starts.sort_unstable_by_key(|&x| Reverse(x));
            starts.dedup();
            let mut steps = cache_steps.iter_mut();
            while let Some(start_target) = starts.pop() {
                loop {
                    let step = steps.next().unwrap();
                    if step.layer == start_target {
                        step.loop_action = Some(LoopAction::Start);
                        break;
                    }
                }
            }
        }

        let last_stateful = cache_steps.iter().rposition(|step| matches!(step.state, StepState::Owned(_))).unwrap();
        let readout_steps = cache_steps
            .split_off(last_stateful + 1)
            .into_iter()
            .map(|step| ReadoutStep {
                layer: step.layer,
                state: match step.state {
                    StepState::Shared(id) => id,
                    StepState::Owned(_) => unreachable!(),
                },
                captures_hidden: step.captures_hidden,
            })
            .collect::<Box<[_]>>();

        let has_readout_hiddens = readout_steps.iter().any(|step| step.captures_hidden);

        Ok(LayerSchedule {
            cache_steps: cache_steps.into_boxed_slice(),
            readout_steps,
            owned_cache_count,
            hidden_capture_count,
            has_readout_hiddens,
        })
    }
}

pub struct TransformerState<B: Backend> {
    /// Each referenced by one [`StateId`].
    states: Box<[Box<dyn MixerState<B>>]>,
    context_length: u32,
}

pub struct TransformerEncodeOutput<B: Backend> {
    pub output: Option<B::ScratchBuffer>,
    pub hidden_features: Box<[B::ScratchBuffer]>,
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
        for state in &mut self.states {
            state.prepare(context_length, suffix_length)?;
        }

        Ok(())
    }

    pub fn encode_accept(
        &mut self,
        batch_dim: &BatchTopology<B>,
        accepted_indices: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        {
            let command_buffer = &mut *command_buffer.debug_group("transformer accept");

            for state in &mut self.states {
                state.encode_accept(batch_dim, accepted_indices, command_buffer)?;
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
    #[error("Hidden features require an additive residual")]
    HiddenFeaturesRequireAdditiveResidual,
    #[error("Layer schedule error: {0}")]
    Schedule(#[from] ScheduleError),
}

#[derive(Debug, Error)]
pub enum TransformerEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Layer error: {0}")]
    Layer(#[from] TransformerLayerError<B>),
    #[error("Residual encoding error: {0}")]
    Residual(#[from] ResidualEncodeError<B>),
}

pub struct Transformer<B: Backend> {
    /// each referenced by one [`LayerId`].
    layers: Box<[TransformerLayer<B>]>,
    schedule: LayerSchedule,
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
        hidden_feature_layer_indices: Option<&[u32]>,
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

        if hidden_feature_layer_indices.is_some() && !matches!(residual_head, ResidualHead::Additive) {
            return Err(TransformerNewError::HiddenFeaturesRequireAdditiveResidual);
        }

        let kv_sources = layers.iter().map(|layer| layer.kv_source_layer_index).collect::<Box<[_]>>();
        let schedule =
            LayerSchedule::new(&kv_sources, transformer_config.layer_order.as_deref(), hidden_feature_layer_indices)?;

        Ok(Self {
            layers: layers.into_boxed_slice(),
            schedule,
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

    fn layer(
        &self,
        id: LayerId,
    ) -> &TransformerLayer<B> {
        &self.layers[id.0 as usize]
    }

    pub fn speculation_supported(&self) -> bool {
        matches!(self.residual_head, ResidualHead::Additive)
            && self.layers.iter().all(|layer| layer.mixer.speculation_supported())
    }

    pub fn max_context_length(&self) -> Option<u32> {
        self.layers
            .iter()
            .map(|layer| layer.mixer.max_context_length())
            .fold(None, |a, b| a.and_then(|a| Some(a.min(b?))).or(a).or(b))
    }

    pub fn prefill_cache_skips_trailing_layers(&self) -> bool {
        !self.schedule.readout_steps.is_empty()
    }

    pub fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<TransformerState<B>, B::Error> {
        let mut states = Vec::with_capacity(self.schedule.owned_cache_count as usize);
        for step in &self.schedule.cache_steps {
            let StepState::Owned(id) = step.state else {
                continue;
            };

            assert_eq!(id.0 as usize, states.len());
            states.push(self.layer(step.layer).mixer.create_empty_state(max_context_length, context)?);
        }

        Ok(TransformerState {
            states: states.into_boxed_slice(),
            context_length: 0,
        })
    }

    pub fn encode(
        &self,
        input: B::ScratchBuffer,
        per_layer_inputs: Option<impl BufferRef<Backend = B>>,
        batch_dim: &BatchTopology<B>,
        output_range: Option<Range<u32>>,
        mut state: Option<&mut TransformerState<B>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<TransformerEncodeOutput<B>, TransformerEncodeError<B>> {
        let mut stream = self
            .residual_head
            .expand(input, batch_dim.size(), command_buffer)
            .map_err(TransformerEncodeError::Backend)?;

        let mut hidden_features = Vec::with_capacity(self.schedule.hidden_capture_count as usize);

        for (step_index, step) in self.schedule.cache_steps.iter().enumerate() {
            let command_buffer =
                &mut *command_buffer.debug_group(&format!("step {step_index} (layer {})", step.layer.0));
            // TODO: `step.loop_action` is available here to handle restarts for block attention
            let layer_state = state.as_mut().map(|state| match step.state {
                StepState::Owned(state_id) => MaybeMut::Mut(state.states[state_id.0 as usize].as_mut()),
                StepState::Shared(state_id) => MaybeMut::Const(state.states[state_id.0 as usize].as_ref()),
            });

            let layer = &self.layers[step.layer.0 as usize];
            stream = layer.encode(stream, per_layer_inputs, batch_dim, layer_state, command_buffer)?;

            if step.captures_hidden {
                let feature =
                    self.capture_hidden_feature(&stream, command_buffer).map_err(TransformerEncodeError::Backend)?;
                hidden_features.push(feature);
            }
        }

        if self.schedule.has_readout_hiddens || output_range.is_some() {
            for (step_index, step) in self.schedule.readout_steps.iter().enumerate() {
                let command_buffer =
                    &mut *command_buffer.debug_group(&format!("readout step {step_index} (layer {})", step.layer.0));
                let layer_state =
                    state.as_ref().map(|state| MaybeMut::Const(state.states[step.state.0 as usize].as_ref()));

                let layer = &self.layers[step.layer.0 as usize];
                stream = layer.encode(stream, per_layer_inputs, batch_dim, layer_state, command_buffer)?;

                if step.captures_hidden {
                    let feature = self
                        .capture_hidden_feature(&stream, command_buffer)
                        .map_err(TransformerEncodeError::Backend)?;
                    hidden_features.push(feature);
                }
            }
        }

        debug_assert_eq!(hidden_features.len(), self.schedule.hidden_capture_count as usize);
        let hidden_features = hidden_features.into_boxed_slice();

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

#[cfg(test)]
#[path = "../../unit/encodable_block/layer_schedule_test.rs"]
mod tests;
