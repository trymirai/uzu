#[cfg(feature = "tracing")]
use std::{cell::RefCell, rc::Rc};
use std::{collections::HashMap, path::Path, time::Instant};

use metal::MTLCommandBuffer;
use objc2::rc::autoreleasepool;

#[cfg(feature = "tracing")]
use super::ActivationTrace;
use super::{ClassificationOutput, ClassificationStats, ClassifierContext};
use crate::{
    DataType,
    backends::metal::Metal,
    encodable_block::{EncodableBlock, EncodingParameters},
    forward_pass::state::{ArrayId, ForwardPassState},
    session::types::Error,
};

pub struct Classifier {
    pub context: ClassifierContext,
}

impl Classifier {
    pub fn new(model_path: &Path) -> Result<Self, Error> {
        let context = ClassifierContext::new(model_path)?;
        Ok(Self {
            context,
        })
    }

    pub fn classify_tokens(
        &mut self,
        token_ids: Vec<u64>,
        token_positions: Vec<usize>,
    ) -> Result<ClassificationOutput, Error> {
        let run_start = Instant::now();

        autoreleasepool(|_pool| {
            let forward_start = Instant::now();

            #[cfg(feature = "tracing")]
            let (logits, _traces) = self.forward_pass_with_traces(&token_ids, &token_positions)?;

            #[cfg(not(feature = "tracing"))]
            let logits = self.forward_pass(&token_ids, &token_positions)?;

            let forward_duration = forward_start.elapsed().as_secs_f64();

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
        })
    }

    #[cfg(feature = "tracing")]
    pub fn forward_pass_with_traces(
        &mut self,
        token_ids: &[u64],
        token_positions: &[usize],
    ) -> Result<(Box<[f32]>, Rc<RefCell<ActivationTrace<Metal>>>), Error> {
        self.forward_pass(token_ids, token_positions)
    }

    #[cfg(feature = "tracing")]
    fn forward_pass(
        &mut self,
        token_ids: &[u64],
        token_positions: &[usize],
    ) -> Result<(Box<[f32]>, Rc<RefCell<ActivationTrace<Metal>>>), Error> {
        autoreleasepool(|_| {
            let num_labels = self.context.model_config.model_config.num_labels;
            let mut state = ForwardPassState::new_classifier(
                self.context.mtl_context.clone(),
                &self.context.model_shape,
                &self.context.scratch_buffers,
                self.context.shared_buffers.clone(),
                token_ids,
                token_positions,
                true,
                num_labels,
            );

            self.context.reset_command_buffer();

            let encoding_params = EncodingParameters::new(false, true, false);

            self.context.embed.encode(&mut state, &encoding_params, &self.context.command_buffer);
            self.context.embedding_norm.encode(&mut state, &encoding_params, &self.context.command_buffer);
            #[cfg(feature = "tracing")]
            {
                let traces = state.traces().clone();
                state.encode_copy_array(
                    &self.context.command_buffer,
                    ArrayId::Main,
                    traces.borrow().embedding_norm().clone(),
                );
            }

            for layer in self.context.layers.iter() {
                layer.encode(&mut state, &encoding_params, &self.context.command_buffer);
            }
            self.context.output_norm.encode(&mut state, &encoding_params, &self.context.command_buffer);
            #[cfg(feature = "tracing")]
            {
                let traces = state.traces().clone();
                state.encode_copy_array(
                    &self.context.command_buffer,
                    ArrayId::Main,
                    traces.borrow().output_norm.clone(),
                );
            }

            self.context.pooling.encode(&mut state, &encoding_params, &self.context.command_buffer);

            self.context.prediction_head.encode(&mut state, &encoding_params, &self.context.command_buffer);

            self.context.command_buffer.commit();
            self.context.command_buffer.wait_until_completed();

            let logits = self.copy_logits_from_state(&state)?;

            let traces = state.traces().clone();
            Ok((logits, traces))
        })
    }

    #[cfg(not(feature = "tracing"))]
    fn forward_pass(
        &mut self,
        token_ids: &[u64],
        token_positions: &[usize],
    ) -> Result<Box<[f32]>, Error> {
        autoreleasepool(|_| {
            let num_labels = self.context.model_config.model_config.num_labels;
            let mut state = ForwardPassState::new_classifier(
                self.context.mtl_context.clone(),
                &self.context.model_shape,
                &self.context.scratch_buffers,
                self.context.shared_buffers.clone(),
                token_ids,
                token_positions,
                true,
                num_labels,
            );

            self.context.reset_command_buffer();

            let encoding_params = EncodingParameters::new(false, true, false);

            self.context.embed.encode(&mut state, &encoding_params, &self.context.command_buffer);
            self.context.embedding_norm.encode(&mut state, &encoding_params, &self.context.command_buffer);
            for layer in self.context.layers.iter() {
                layer.encode(&mut state, &encoding_params, &self.context.command_buffer);
            }
            self.context.output_norm.encode(&mut state, &encoding_params, &self.context.command_buffer);
            self.context.pooling.encode(&mut state, &encoding_params, &self.context.command_buffer);
            self.context.prediction_head.encode(&mut state, &encoding_params, &self.context.command_buffer);

            self.context.command_buffer.commit();
            self.context.command_buffer.wait_until_completed();

            let logits = self.copy_logits_from_state(&state)?;

            Ok(logits)
        })
    }

    fn copy_logits_from_state(
        &self,
        state: &ForwardPassState<Metal>,
    ) -> Result<Box<[f32]>, Error> {
        let logits_arrays = state.arrays(&[ArrayId::ClassifierPredictionHeadLogits]);
        let logits_array = logits_arrays[0].borrow();

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
