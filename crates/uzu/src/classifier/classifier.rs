use std::{collections::HashMap, path::Path, time::Instant};

#[cfg(feature = "tracing")]
use super::ActivationTrace;
use super::{ClassificationOutput, ClassificationStats, ClassifierContext};
use crate::{
    DataType,
    array::allocation_as_slice,
    backends::common::{Allocation, Backend, Encoder},
    config::ModelMetadata,
    encodable_block::{EncodingParameters, LayerArguments},
    forward_pass::state::ForwardPassState,
    session::types::Error,
};

pub struct Classifier<B: Backend> {
    pub context: ClassifierContext<B>,
}

pub trait ClassifierTrait {
    fn classify_tokens(
        &mut self,
        token_ids: Vec<u64>,
        token_positions: Vec<usize>,
    ) -> Result<ClassificationOutput, Error>;
}

impl<B: Backend> ClassifierTrait for Classifier<B> {
    fn classify_tokens(
        &mut self,
        token_ids: Vec<u64>,
        token_positions: Vec<usize>,
    ) -> Result<ClassificationOutput, Error> {
        let run_start = Instant::now();

        let logits = self.forward_pass(&token_ids, &token_positions)?;

        let forward_duration = run_start.elapsed().as_secs_f64();

        let postprocessing_start = Instant::now();
        let probabilities = self.logits_to_probabilities(&logits)?;
        let postprocessing_duration = postprocessing_start.elapsed().as_secs_f64();

        let (predicted_label, confidence) = probabilities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(label, prob)| (label.clone(), *prob))
            .unwrap_or((String::from("unknown"), 0.0));

        let stats = ClassificationStats::new(
            0.0,
            forward_duration,
            postprocessing_duration,
            run_start.elapsed().as_secs_f64(),
            token_ids.len() as u64,
            predicted_label,
            confidence,
        );

        Ok(ClassificationOutput {
            logits,
            probabilities,
            stats,
        })
    }
}

impl<B: Backend> Classifier<B> {
    pub fn new(
        model_path: &Path,
        model_metadata: &ModelMetadata,
    ) -> Result<Self, Error> {
        let context = ClassifierContext::new(model_path, model_metadata)?;
        Ok(Self {
            context,
        })
    }

    #[cfg(feature = "tracing")]
    pub fn forward_pass_with_traces(
        &mut self,
        token_ids: &[u64],
        token_positions: &[usize],
    ) -> Result<(Box<[f32]>, ActivationTrace<B>), Error> {
        let num_labels = self.context.model_config.model_config.num_labels;
        let traces = ActivationTrace::new_classifier(
            self.context.context.as_ref(),
            &self.context.model_shape,
            token_ids.len(),
            num_labels,
        );
        let logits = self.forward_pass_impl(token_ids, token_positions, Some(&traces))?;
        Ok((logits, traces))
    }

    #[cfg(feature = "tracing")]
    fn forward_pass(
        &mut self,
        token_ids: &[u64],
        token_positions: &[usize],
    ) -> Result<Box<[f32]>, Error> {
        self.forward_pass_impl(token_ids, token_positions, None)
    }

    fn forward_pass_impl(
        &mut self,
        token_ids: &[u64],
        token_positions: &[usize],
        #[cfg(feature = "tracing")] trace: Option<&ActivationTrace<B>>,
    ) -> Result<Box<[f32]>, Error> {
        let state = ForwardPassState::new_classifier(
            self.context.context.clone(),
            &self.context.model_shape,
            self.context.shared_buffers.clone(),
            token_ids,
            token_positions,
        );

        let encoding_params = EncodingParameters::new();

        let mut encoder = Encoder::<B>::new(self.context.context.as_ref())
            .map_err(|e| Error::UnableToCreateCommandBuffer(e.into()))?;

        let mut main = self
            .context
            .embed
            .encode_lookup(
                state.token_ids(),
                state.active_row_count(),
                state.model_shape().activation_data_type(),
                &mut encoder,
            )
            .map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        main = self
            .context
            .embedding_norm
            .encode(&main, 0, state.active_row_count(), &mut encoder)
            .map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        #[cfg(feature = "tracing")]
        if let Some(trace) = trace {
            encoder.encode_copy_allocation(&main, trace.embedding_norm());
        }

        let mut shortcut =
            encoder.allocate_scratch(main.as_buffer_range().1.len()).map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        for (layer_index, layer) in self.context.layers.iter().enumerate() {
            main = layer
                .encode(
                    LayerArguments {
                        context: state.context(),
                        batch_dim: state.active_row_count(),
                        token_positions: state.token_positions(),
                        token_parents: state.token_parents(),
                        token_subtrie_ranges: state.token_subtrie_ranges(),
                        attention_sinks: state.attention_sinks(layer_index),
                        rope_cosines: Some(state.rope_cosines(layer.rope_type())),
                        rope_sines: Some(state.rope_sines(layer.rope_type())),
                        rope_max_sequence_length: state.rope_max_sequence_length(),
                        rope_dim: state.rope_dim(),
                        sampling_start: state.sampling_start(),
                        sampling_length: state.sampling_length(),
                        cache_layer: None,
                        #[cfg(feature = "tracing")]
                        trace: trace.and_then(|traces| traces.layer_results.get(layer_index)),
                    },
                    &encoding_params,
                    main,
                    &mut shortcut,
                    &mut encoder,
                )
                .map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        }
        main = self
            .context
            .output_norm
            .encode(&main, 0, state.active_row_count(), &mut encoder)
            .map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        #[cfg(feature = "tracing")]
        if let Some(trace) = trace {
            encoder.encode_copy_allocation(&main, &trace.output_norm);
        }
        let pooling = self
            .context
            .pooling
            .encode(state.active_row_count(), &main, &mut encoder)
            .map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        #[cfg(feature = "tracing")]
        if let Some(trace) = trace {
            encoder.encode_copy_allocation(&pooling, trace.output_pooling());
        }
        let logits = self
            .context
            .prediction_head
            .encode(state.context(), &pooling, &mut encoder)
            .map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        #[cfg(feature = "tracing")]
        if let Some(trace) = trace {
            encoder.encode_copy_allocation(&logits, &trace.logits);
        }

        let _completed = encoder
            .end_encoding()
            .submit()
            .wait_until_completed()
            .map_err(|e| Error::CommandBufferFailed(Box::new(e)))?;

        let logits = self.copy_logits_from_allocation(&logits)?;

        Ok(logits)
    }

    #[cfg(not(feature = "tracing"))]
    fn forward_pass(
        &mut self,
        token_ids: &[u64],
        token_positions: &[usize],
    ) -> Result<Box<[f32]>, Error> {
        self.forward_pass_impl(token_ids, token_positions)
    }

    fn copy_logits_from_allocation(
        &self,
        logits: &Allocation<B>,
    ) -> Result<Box<[f32]>, Error> {
        let num_labels = self.context.model_config.model_config.num_labels;
        let logits_data_type: DataType =
            self.context.model_config.model_config.prediction_head_config.readout_config.activation_precision().into();

        match logits_data_type {
            DataType::F32 => Ok(allocation_as_slice::<f32, B>(logits).iter().copied().take(num_labels).collect()),
            DataType::F16 => Ok(allocation_as_slice::<half::f16, B>(logits)
                .into_iter()
                .take(num_labels)
                .map(|x| x.to_f32())
                .collect::<Box<[_]>>()),
            DataType::BF16 => Ok(allocation_as_slice::<half::bf16, B>(logits)
                .into_iter()
                .take(num_labels)
                .map(|x| x.to_f32())
                .collect::<Box<[_]>>()),
            _ => Err(Error::UnableToDecodeText),
        }
    }

    fn logits_to_probabilities(
        &self,
        logits: &[f32],
    ) -> Result<HashMap<String, f32>, Error> {
        let output_labels = &self.context.model_config.model_config.output_labels;
        let mut probabilities = HashMap::new();

        for (idx, &logit) in logits.iter().enumerate() {
            let prob = 1.0 / (1.0 + (-logit).exp());

            let label = if let Some(labels) = output_labels {
                labels.get(idx).map(|s| s.clone()).unwrap_or_else(|| format!("class_{}", idx))
            } else {
                format!("class_{}", idx)
            };

            probabilities.insert(label, prob);
        }

        Ok(probabilities)
    }
}
