use std::{collections::HashMap, path::Path, time::Instant};

#[cfg(feature = "tracing")]
use super::ActivationTrace;
use super::{ClassificationOutput, ClassificationStats, ClassifierContext};
use crate::{
    DataType, allocation_to_vec,
    backends::common::{Allocation, AsBufferRangeRef, Backend, Encoder},
    config::model::classifier_model::ClassifierModelConfig,
    encodable_block::LayerArguments,
    forward_pass::token_inputs::TokenInputs,
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
        model_config: &ClassifierModelConfig,
    ) -> Result<Self, Error> {
        let context = ClassifierContext::new(model_path, model_config)?;
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
        let num_labels = self.context.model_config.classifier_config.num_labels;
        let mut traces = ActivationTrace::new_classifier(
            self.context.context.as_ref(),
            &self.context.model_shape,
            token_ids.len(),
            num_labels,
        );
        let logits = self.forward_pass_impl(token_ids, token_positions, Some(&mut traces))?;
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
        #[cfg(feature = "tracing")] trace: Option<&mut ActivationTrace<B>>,
    ) -> Result<Box<[f32]>, Error> {
        let batch_dim = token_ids.len();
        let token_inputs = TokenInputs::new_classifier(self.context.context.as_ref(), token_ids, token_positions);

        #[cfg(feature = "tracing")]
        let mut trace = trace;
        let mut encoder = Encoder::<B>::new(self.context.context.as_ref())
            .map_err(|e| Error::UnableToCreateCommandBuffer(e.into()))?;

        let mut main = self
            .context
            .embed
            .encode_lookup(token_inputs.token_ids(), batch_dim, &mut encoder)
            .map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        main = self
            .context
            .embedding_norm
            .encode(&main, 0, batch_dim, &mut encoder)
            .map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        #[cfg(feature = "tracing")]
        if let Some(trace) = trace.as_deref_mut() {
            let embedding_norm = trace.embedding_norm_mut().allocation_mut();
            encoder.encode_copy(&main, .., embedding_norm, ..);
        }

        let mut shortcut = encoder
            .allocate_scratch(main.as_buffer_range_ref().range().len())
            .map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        for (layer_index, layer) in self.context.layers.iter().enumerate() {
            let rope_buffers = self.context.shared_buffers.rope_buffers_for_layer(layer_index);
            main = layer
                .encode(
                    LayerArguments {
                        batch_dim,
                        token_positions: token_inputs.token_positions(),
                        token_parents: token_inputs.token_parents(),
                        token_subtrie_ranges: None,
                        rope_buffers,
                        sampling_start: 0,
                        sampling_length: batch_dim,
                        cache_access: None,
                        #[cfg(feature = "tracing")]
                        trace: trace.as_deref_mut().map(|traces| &mut traces.layer_results[layer_index]),
                    },
                    main,
                    &mut shortcut,
                    &mut encoder,
                )
                .map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        }
        main = self
            .context
            .output_norm
            .encode(&main, 0, batch_dim, &mut encoder)
            .map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        #[cfg(feature = "tracing")]
        if let Some(trace) = trace.as_deref_mut() {
            encoder.encode_copy(&main, .., trace.output_norm.allocation_mut(), ..);
        }
        let pooling = self
            .context
            .pooling
            .encode(batch_dim, &main, &mut encoder)
            .map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        #[cfg(feature = "tracing")]
        if let Some(trace) = trace.as_deref_mut() {
            let output_pooling = trace.output_pooling_mut().allocation_mut();
            encoder.encode_copy(&pooling, .., output_pooling, ..);
        }
        let logits =
            self.context.prediction_head.encode(pooling, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        #[cfg(feature = "tracing")]
        if let Some(trace) = trace.as_deref_mut() {
            encoder.encode_copy(&logits, .., trace.logits.allocation_mut(), ..);
        }

        let completed = encoder
            .end_encoding()
            .submit()
            .wait_until_completed()
            .map_err(|e| Error::CommandBufferFailed(Box::new(e)))?;

        let logits_result = self.copy_logits_from_allocation(&logits);
        drop(logits);
        drop(main);
        drop(shortcut);
        drop(completed);

        logits_result
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
        match self.context.logits_data_type {
            DataType::F32 => Ok(allocation_to_vec::<B, f32>(logits).into()),
            DataType::F16 => {
                Ok(allocation_to_vec::<B, half::f16>(logits).into_iter().map(|x| x.to_f32()).collect::<Box<[_]>>())
            },
            DataType::BF16 => {
                Ok(allocation_to_vec::<B, half::bf16>(logits).into_iter().map(|x| x.to_f32()).collect::<Box<[_]>>())
            },
            data_type => {
                Err(Error::InvalidModelConfig(format!("unsupported classifier logits data type {data_type:?}")))
            },
        }
    }

    fn logits_to_probabilities(
        &self,
        logits: &[f32],
    ) -> Result<HashMap<String, f32>, Error> {
        let output_labels = &self.context.model_config.classifier_config.output_labels;
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
