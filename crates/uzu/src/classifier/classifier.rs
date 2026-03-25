#[cfg(feature = "tracing")]
use std::{cell::RefCell, rc::Rc};
use std::{collections::HashMap, path::Path, time::Instant};

#[cfg(feature = "tracing")]
use super::ActivationTrace;
use super::{ClassificationOutput, ClassificationStats, ClassifierContext};
use crate::{
    DataType,
    backends::common::{Backend, Encoder},
    config::ModelMetadata,
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
        let (logits, _traces) = self.forward_pass_with_traces(&token_ids, &token_positions)?;

        #[cfg(not(feature = "tracing"))]
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
    ) -> Result<(Box<[f32]>, Rc<RefCell<ActivationTrace<B>>>), Error> {
        self.forward_pass(token_ids, token_positions)
    }

    #[cfg(feature = "tracing")]
    fn forward_pass(
        &mut self,
        token_ids: &[u64],
        token_positions: &[usize],
    ) -> Result<(Box<[f32]>, Rc<RefCell<ActivationTrace<B>>>), Error> {
        let num_labels = self.context.model_config.model_config.num_labels;
        let mut state = ForwardPassState::new_classifier(
            self.context.context.clone(),
            &self.context.model_shape,
            &self.context.scratch_buffers,
            self.context.shared_buffers.clone(),
            token_ids,
            token_positions,
            true,
            num_labels,
        );

        let encoding_params = EncodingParameters::new();

        let mut encoder = Encoder::<B>::new(self.context.context.as_ref())
            .map_err(|e| Error::UnableToCreateCommandBuffer(e.into()))?;

        self.context.embed.encode_lookup(&mut state, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        self.context.embedding_norm.encode(&mut state, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;
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

        let logits = self.copy_logits_from_state(&state)?;

        let traces = state.traces().clone();
        Ok((logits, traces))
    }

    #[cfg(not(feature = "tracing"))]
    fn forward_pass(
        &mut self,
        token_ids: &[u64],
        token_positions: &[usize],
    ) -> Result<Box<[f32]>, Error> {
        let num_labels = self.context.model_config.model_config.num_labels;
        let mut state = ForwardPassState::new_classifier(
            self.context.context.clone(),
            &self.context.model_shape,
            &self.context.scratch_buffers,
            self.context.shared_buffers.clone(),
            token_ids,
            token_positions,
            true,
            num_labels,
        );

        let encoding_params = EncodingParameters::new();

        let mut encoder = Encoder::<B>::new(self.context.context.as_ref())
            .map_err(|e| Error::UnableToCreateCommandBuffer(e.into()))?;

        self.context.embed.encode_lookup(&mut state, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        self.context.embedding_norm.encode(&mut state, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        for layer in self.context.layers.iter() {
            layer.encode(&mut state, &encoding_params, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        }
        self.context.output_norm.encode(&mut state, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        self.context.pooling.encode(&mut state, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        self.context.prediction_head.encode(&mut state, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;

        encoder.end_encoding().submit().wait_until_completed().map_err(|e| Error::CommandBufferFailed(Box::new(e)))?;

        let logits = self.copy_logits_from_state(&state)?;

        Ok(logits)
    }

    fn copy_logits_from_state(
        &self,
        state: &ForwardPassState<B>,
    ) -> Result<Box<[f32]>, Error> {
        let logits_arrays = state.arrays(&[ArrayId::ClassifierPredictionHeadLogits]);
        let logits_array = &logits_arrays[0];

        let num_labels = self.context.model_config.model_config.num_labels;
        let buffer = logits_array.as_bytes();

        match logits_array.data_type() {
            DataType::F32 => {
                let slice: &[f32] = bytemuck::cast_slice(buffer);
                Ok(slice[..num_labels].into())
            },
            DataType::F16 => {
                let slice: &[half::f16] = bytemuck::cast_slice(buffer);
                Ok(slice[..num_labels].iter().map(|&x| x.to_f32()).collect::<Vec<_>>().into_boxed_slice())
            },
            DataType::BF16 => {
                let slice: &[half::bf16] = bytemuck::cast_slice(buffer);
                Ok(slice[..num_labels].iter().map(|&x| x.to_f32()).collect::<Vec<_>>().into_boxed_slice())
            },
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
