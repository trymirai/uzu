#[cfg(feature = "tracing")]
use std::{cell::RefCell, rc::Rc};
use std::{collections::HashMap, path::Path, time::Instant};

#[cfg(feature = "tracing")]
use super::ActivationTrace;
use super::{ClassificationOutput, ClassificationStats, ClassifierContext};
use crate::{
    DataType,
    backends::common::{Backend, Encoder},
    config::{ModelMetadata, PoolingType},
    encodable_block::EncodingParameters,
    forward_pass::state::{ArrayId, ForwardPassState},
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

        #[cfg(feature = "tracing")]
        let (logits, num_rows, _traces) = self.forward_pass_with_traces(&token_ids, &token_positions)?;

        #[cfg(not(feature = "tracing"))]
        let (logits, num_rows) = self.forward_pass(&token_ids, &token_positions)?;

        let forward_duration = run_start.elapsed().as_secs_f64();
        let num_labels = self.context.model_config.model_config.num_labels;

        let postprocessing_start = Instant::now();
        let per_token_mode = matches!(
            self.context.model_config.model_config.classifier_pooling,
            PoolingType::None
        );

        let (probabilities, per_token_top1, predicted_label, confidence) = if per_token_mode {
            let tokens = self.logits_to_per_token_top1(&logits, num_rows, num_labels);
            // Report the max-confidence token as the top-level "prediction" so
            // existing callers still get *something* meaningful. The full
            // per-token sequence lives in `per_token_top1`.
            let (predicted_label, confidence) = tokens
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap())
                .map(|(idx, (label, prob))| (format!("tok{}:{}", idx, label), *prob))
                .unwrap_or_else(|| (String::from("unknown"), 0.0));
            (HashMap::new(), Some(tokens), predicted_label, confidence)
        } else {
            let probabilities = self.logits_to_probabilities(&logits)?;
            let (predicted_label, confidence) = probabilities
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(label, prob)| (label.clone(), *prob))
                .unwrap_or((String::from("unknown"), 0.0));
            (probabilities, None, predicted_label, confidence)
        };
        let postprocessing_duration = postprocessing_start.elapsed().as_secs_f64();

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
            num_rows,
            num_labels,
            probabilities,
            per_token_top1,
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
    ) -> Result<(Box<[f32]>, usize, Rc<RefCell<ActivationTrace<B>>>), Error> {
        self.forward_pass(token_ids, token_positions)
    }

    #[cfg(feature = "tracing")]
    fn forward_pass(
        &mut self,
        token_ids: &[u64],
        token_positions: &[usize],
    ) -> Result<(Box<[f32]>, usize, Rc<RefCell<ActivationTrace<B>>>), Error> {
        let num_labels = self.context.model_config.model_config.num_labels;
        let mut state = ForwardPassState::new_classifier(
            self.context.context.clone(),
            self.context.decoder_config.as_ref(),
            &self.context.model_shape,
            &self.context.scratch_buffers,
            self.context.shared_buffers.clone(),
            token_ids,
            token_positions,
            num_labels,
        );

        let encoding_params = EncodingParameters::new();

        let mut encoder = Encoder::<B>::new(self.context.context.as_ref())
            .map_err(|e| Error::UnableToCreateCommandBuffer(e.into()))?;

        self.context.embed.encode_lookup(&mut state, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        if let Some(embedding_norm) = self.context.embedding_norm.as_ref() {
            embedding_norm.encode(&mut state, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        }
        #[cfg(feature = "tracing")]
        {
            let traces = state.traces().clone();
            state.encode_copy_array(&mut encoder, ArrayId::Main, traces.borrow().embedding_norm().clone());
        }

        for layer in self.context.layers.iter() {
            layer.encode(&mut state, &encoding_params, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        }
        self.context.output_norm.encode(&mut state, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        #[cfg(feature = "tracing")]
        {
            let traces = state.traces().clone();
            state.encode_copy_array(&mut encoder, ArrayId::Main, traces.borrow().output_norm.clone());
        }

        self.context.pooling.encode(&mut state, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;

        self.context.prediction_head.encode(&mut state, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;

        encoder.end_encoding().submit().wait_until_completed().map_err(|e| Error::CommandBufferFailed(Box::new(e)))?;

        let (logits, num_rows) = self.copy_logits_from_state(&state)?;

        let traces = state.traces().clone();
        Ok((logits, num_rows, traces))
    }

    #[cfg(not(feature = "tracing"))]
    fn forward_pass(
        &mut self,
        token_ids: &[u64],
        token_positions: &[usize],
    ) -> Result<(Box<[f32]>, usize), Error> {
        let num_labels = self.context.model_config.model_config.num_labels;
        let mut state = ForwardPassState::new_classifier(
            self.context.context.clone(),
            self.context.decoder_config.as_ref(),
            &self.context.model_shape,
            &self.context.scratch_buffers,
            self.context.shared_buffers.clone(),
            token_ids,
            token_positions,
            num_labels,
        );

        let encoding_params = EncodingParameters::new();

        let mut encoder = Encoder::<B>::new(self.context.context.as_ref())
            .map_err(|e| Error::UnableToCreateCommandBuffer(e.into()))?;

        self.context.embed.encode_lookup(&mut state, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        if let Some(embedding_norm) = self.context.embedding_norm.as_ref() {
            embedding_norm.encode(&mut state, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        }
        for layer in self.context.layers.iter() {
            layer.encode(&mut state, &encoding_params, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        }
        self.context.output_norm.encode(&mut state, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        self.context.pooling.encode(&mut state, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        self.context.prediction_head.encode(&mut state, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;

        encoder.end_encoding().submit().wait_until_completed().map_err(|e| Error::CommandBufferFailed(Box::new(e)))?;

        self.copy_logits_from_state(&state)
    }

    fn copy_logits_from_state(
        &self,
        state: &ForwardPassState<B>,
    ) -> Result<(Box<[f32]>, usize), Error> {
        let num_labels = self.context.model_config.model_config.num_labels;
        let per_token = matches!(
            self.context.model_config.model_config.classifier_pooling,
            PoolingType::None
        );
        // Pooling collapses to a single row; per-token readout produces
        // `active_row_count` rows.
        let num_rows = if per_token { state.active_row_count() } else { 1 };
        let logits = state.array(ArrayId::ClassifierPredictionHeadLogits).view(&[num_rows, num_labels]);

        let flat: Box<[f32]> = match logits.data_type() {
            DataType::F32 => logits.as_slice::<f32>().into(),
            DataType::F16 => logits.as_slice::<half::f16>().iter().map(|&x| x.to_f32()).collect::<Box<[_]>>(),
            DataType::BF16 => logits.as_slice::<half::bf16>().iter().map(|&x| x.to_f32()).collect::<Box<[_]>>(),
            _ => return Err(Error::UnableToDecodeText),
        };
        Ok((flat, num_rows))
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

    /// Per-token argmax + softmax-confidence. Assumes `logits` is row-major
    /// `[num_rows, num_labels]`.
    fn logits_to_per_token_top1(
        &self,
        logits: &[f32],
        num_rows: usize,
        num_labels: usize,
    ) -> Vec<(String, f32)> {
        let output_labels = &self.context.model_config.model_config.output_labels;
        let mut out = Vec::with_capacity(num_rows);
        for row in 0..num_rows {
            let start = row * num_labels;
            let end = start + num_labels;
            let row_logits = &logits[start..end];

            // argmax
            let (best_idx, &best_logit) = row_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap_or((0, &0.0));

            // numerically stable softmax-confidence for the argmax row.
            let max_logit = row_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for &l in row_logits.iter() {
                sum_exp += (l - max_logit).exp();
            }
            let conf = if sum_exp > 0.0 { (best_logit - max_logit).exp() / sum_exp } else { 0.0 };

            let label = if let Some(labels) = output_labels {
                labels.get(best_idx).map(|s| s.clone()).unwrap_or_else(|| format!("class_{}", best_idx))
            } else {
                format!("class_{}", best_idx)
            };
            out.push((label, conf));
        }
        out
    }
}
