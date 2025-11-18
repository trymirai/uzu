use std::{cell::RefCell, path::Path, rc::Rc, time::Instant};

use objc2::rc::autoreleasepool;

use super::{
    ClassificationForwardPassState, ClassificationOutput, ClassificationStats,
    ClassifierContext,
};
use crate::{
    DataType,
    backends::metal::{
        MetalArray,
        forward_pass::{
            ArrayId, ForwardPassState,
            encodable_with_state::{EncodableWithState, EncodingParameters},
        },
    },
    config::PoolingType,
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

            let (logits, _traces) =
                self.forward_pass(&token_ids, &token_positions, false)?;

            let forward_duration = forward_start.elapsed().as_secs_f64();
            let probabilities = self.logits_to_probabilities(&logits)?;

            let stats = ClassificationStats::new(
                forward_duration,
                run_start.elapsed().as_secs_f64(),
            );

            Ok(ClassificationOutput {
                logits,
                probabilities,
                stats,
            })
        })
    }

    pub fn forward_pass_with_traces(
        &mut self,
        token_ids: &[u64],
        token_positions: &[usize],
    ) -> Result<(Vec<f32>, Rc<RefCell<super::ClassifierActivationTrace>>), Error>
    {
        self.forward_pass(token_ids, token_positions, true)
    }

    fn forward_pass(
        &mut self,
        token_ids: &[u64],
        token_positions: &[usize],
        enable_traces: bool,
    ) -> Result<(Vec<f32>, Rc<RefCell<super::ClassifierActivationTrace>>), Error>
    {
        autoreleasepool(|_| {
            // Create ClassificationForwardPassState with bidirectional attention
            // No KV cache, no vocabulary-sized buffers - designed for classification
            let num_labels =
                self.context.model_config.classifier_config.num_labels;
            let mut state = ClassificationForwardPassState::new(
                self.context.mtl_context.clone(),
                &self
                    .context
                    .model_config
                    .classifier_config
                    .to_decoder_config(),
                &self.context.model_shape,
                &self.context.scratch_buffers,
                self.context.shared_buffers.clone(),
                token_ids,
                token_positions,
                true, // Bidirectional attention for BERT
                enable_traces,
                num_labels,
            );

            let encoding_params = EncodingParameters::new(false, true, false);

            // Reset command buffer for this forward pass
            self.context.reset_command_buffer();

            self.context.embed.encode(
                &mut state,
                &self.context.command_buffer,
                &encoding_params,
            );
            self.context.embedding_norm.encode(
                &mut state,
                &self.context.command_buffer,
                &encoding_params,
            );
            // Trace: embedding_norm
            if enable_traces {
                if let Some(traces) = state.classifier_traces().cloned() {
                    state.encode_copy_array(
                        &self.context.command_buffer,
                        ArrayId::Main,
                        traces.borrow().embedding_norm.clone(),
                    );
                }
            }

            for layer in self.context.layers.iter() {
                layer.encode(
                    &mut state,
                    &self.context.command_buffer,
                    &encoding_params,
                );

                // If tracing is enabled, commit and wait after each layer
                // to ensure trace copies complete before next layer starts
                if enable_traces {
                    let root = self
                        .context
                        .command_buffer
                        .root_command_buffer()
                        .to_owned();
                    self.context.command_buffer.commit_and_continue();
                    root.wait_until_completed();
                }
            }
            self.context.output_norm.encode(
                &mut state,
                &self.context.command_buffer,
                &encoding_params,
            );
            // Trace: output_norm
            if enable_traces {
                if let Some(traces) = state.classifier_traces().cloned() {
                    state.encode_copy_array(
                        &self.context.command_buffer,
                        ArrayId::Main,
                        traces.borrow().output_norm.clone(),
                    );
                }
            }

            // Apply pooling: [batch, seq_len, model_dim] → [batch, model_dim]
            let mut pooled_output = self.apply_pooling(&mut state)?;

            // Apply prediction head: [batch, model_dim] → [batch, num_labels]
            self.apply_prediction_head(&mut state, &mut pooled_output)?;

            // Non-trace path synchronization is handled inside apply_prediction_head

            // Copy result from GPU to CPU (Array::buffer() will implicitly sync)
            let logits = self.copy_logits_from_state(&state)?;

            let traces = if enable_traces {
                state
                    .classifier_traces()
                    .expect("Traces should be enabled")
                    .clone()
            } else {
                // Allocate a minimal non-zero-sized traces object to avoid zero-sized Metal buffers
                Rc::new(RefCell::new(super::ClassifierActivationTrace::new(
                    &self.context.mtl_context,
                    &self.context.model_shape,
                    1,
                    1,
                )))
            };

            Ok((logits, traces))
        })
    }

    fn apply_pooling(
        &mut self,
        state: &mut ClassificationForwardPassState,
    ) -> Result<MetalArray, Error> {
        let batch_size = 1;
        let seq_len = state.aux_buffers_suffix_length();
        let model_dim = self.context.model_config.classifier_config.model_dim;

        let arrays = state.arrays(&[ArrayId::Main, ArrayId::ClassifierPooling]);
        let data_type = {
            use crate::Array;
            Array::data_type(&*arrays[0].borrow())
        };
        let mut main_array = arrays[0].borrow_mut();
        let input_buffer = unsafe { main_array.mtl_buffer() };

        // Use pre-allocated pooling buffer from state
        let mut pooling_array = arrays[1].borrow_mut();
        let output_buffer = unsafe { pooling_array.mtl_buffer().to_owned() };

        let root = self.context.command_buffer.root_command_buffer();
        let encoder = root.new_compute_command_encoder();

        match self.context.model_config.classifier_config.classifier_pooling {
            PoolingType::Cls => {
                self.context
                    .pooling_kernel
                    .encode_cls(
                        encoder,
                        input_buffer,
                        &output_buffer,
                        batch_size as i32,
                        seq_len as i32,
                        model_dim as i32,
                    )
                    .map_err(|_| Error::PrefillFailed)?;
            },
            PoolingType::Mean => {
                self.context
                    .pooling_kernel
                    .encode_mean(
                        encoder,
                        input_buffer,
                        &output_buffer,
                        batch_size as i32,
                        seq_len as i32,
                        model_dim as i32,
                    )
                    .map_err(|_| Error::PrefillFailed)?;
            },
        }

        encoder.end_encoding();
        drop(main_array);

        self.context.command_buffer.commit_and_continue();

        // Trace: output_pooling
        if let Some(traces_rc) = state.classifier_traces().cloned() {
            let traces_ref = traces_rc.borrow();
            let mut trace_arr = traces_ref.output_pooling.borrow_mut();
            let dst_buf = unsafe { trace_arr.mtl_buffer().to_owned() };
            drop(trace_arr);
            drop(traces_ref);

            let root = self.context.command_buffer.root_command_buffer();
            let blit = root.new_blit_command_encoder();
            blit.copy_from_buffer(
                &output_buffer,
                0,
                &dst_buf,
                0,
                (batch_size * model_dim * data_type.size_in_bytes()) as u64,
            );
            blit.end_encoding();

            let root_owned = root.to_owned();
            self.context.command_buffer.commit_and_continue();
            root_owned.wait_until_completed();
        }

        // Return the pooling array directly (already has correct buffer/shape/dtype)
        Ok(pooling_array.clone())
    }

    fn apply_prediction_head(
        &mut self,
        state: &mut ClassificationForwardPassState,
        pooled_input: &mut MetalArray,
    ) -> Result<(), Error> {
        let batch_size = {
            use crate::Array;
            Array::shape(pooled_input)[0]
        };
        let num_labels = self.context.model_config.classifier_config.num_labels;
        let data_type = {
            use crate::Array;
            Array::data_type(pooled_input)
        };

        // Copy pooled input into ClassifierPredictionHeadPooled buffer (GPU blit, in-order)
        {
            let arrays = state.arrays(&[
                ArrayId::ClassifierPredictionHeadPooled,
                ArrayId::ClassifierPooling,
            ]);
            let mut dst = arrays[0].borrow_mut();
            let dst_buf = unsafe { dst.mtl_buffer().to_owned() };
            drop(dst);
            let mut src = arrays[1].borrow_mut();
            let src_buf = unsafe { src.mtl_buffer().to_owned() };
            drop(src);

            let copy_size_bytes = (batch_size
                * self.context.model_config.classifier_config.model_dim
                * data_type.size_in_bytes())
                as u64;

            let root = self.context.command_buffer.root_command_buffer();
            let blit = root.new_blit_command_encoder();
            blit.copy_from_buffer(&src_buf, 0, &dst_buf, 0, copy_size_bytes);
            blit.end_encoding();
            self.context.command_buffer.commit_and_continue();
        }

        let encoding_params = EncodingParameters::new(false, true, false);

        // RUN ALL 4 PREDICTION HEAD OPERATIONS using the single state
        // Pipeline: ClassifierPredictionHeadPooled → Dense → Norm → Logits

        // Step 1: Dense (Pooled → Dense)
        self.context.prediction_head_dense.encode(
            state,
            &self.context.command_buffer,
            &encoding_params,
        );

        // Step 2: GELU (Dense → Dense, in-place)
        self.context.prediction_head_activation.encode(
            state,
            &self.context.command_buffer,
            &encoding_params,
        );

        // Step 3: Norm (Dense → Norm)
        self.context.prediction_head_norm.encode(
            state,
            &self.context.command_buffer,
            &encoding_params,
        );

        // Step 4: Final linear (Norm → Logits)
        self.context.prediction_head_final_linear.encode(
            state,
            &self.context.command_buffer,
            &encoding_params,
        );

        // Commit and optionally wait here for non-trace path.
        let root_after_head =
            self.context.command_buffer.root_command_buffer().to_owned();
        self.context.command_buffer.commit_and_continue();
        if state.classifier_traces().is_none() {
            root_after_head.wait_until_completed();
        }

        // Trace: logits (copy linear outputs to trace buffer)
        if let Some(traces_rc) = state.classifier_traces().cloned() {
            let logits_arrays =
                state.arrays(&[ArrayId::ClassifierPredictionHeadLogits]);
            let mut logits_array_ref = logits_arrays[0].borrow_mut();
            let linear_output_buffer =
                unsafe { logits_array_ref.mtl_buffer().to_owned() };
            drop(logits_array_ref);

            let traces_ref = traces_rc.borrow();
            let mut trace_logits = traces_ref.logits.borrow_mut();
            let dst_trace_buf = unsafe { trace_logits.mtl_buffer().to_owned() };
            drop(trace_logits);
            drop(traces_ref);

            let copy_size_bytes =
                (batch_size * num_labels * data_type.size_in_bytes()) as u64;

            let root = self.context.command_buffer.root_command_buffer();
            let blit = root.new_blit_command_encoder();
            blit.copy_from_buffer(
                &linear_output_buffer,
                0,
                &dst_trace_buf,
                0,
                copy_size_bytes,
            );
            blit.end_encoding();

            let root_owned = root.to_owned();
            self.context.command_buffer.commit_and_continue();
            root_owned.wait_until_completed();
        }

        Ok(())
    }

    fn copy_logits_from_state(
        &self,
        state: &ClassificationForwardPassState,
    ) -> Result<Vec<f32>, Error> {
        let logits_arrays =
            state.arrays(&[ArrayId::ClassifierPredictionHeadLogits]);
        let logits_array = logits_arrays[0].borrow();

        use crate::Array;
        let num_labels = self.context.model_config.classifier_config.num_labels;
        let buffer = Array::buffer(&*logits_array);

        match Array::data_type(&*logits_array) {
            DataType::F32 => {
                let slice: &[f32] = bytemuck::cast_slice(buffer);
                Ok(slice[..num_labels].to_vec())
            },
            DataType::F16 => {
                let slice: &[half::f16] = bytemuck::cast_slice(buffer);
                Ok(slice[..num_labels].iter().map(|&x| x.to_f32()).collect())
            },
            DataType::BF16 => {
                let slice: &[half::bf16] = bytemuck::cast_slice(buffer);
                Ok(slice[..num_labels].iter().map(|&x| x.to_f32()).collect())
            },
            _ => Err(Error::UnableToDecodeText),
        }
    }

    fn logits_to_probabilities(
        &self,
        logits: &[f32],
    ) -> Result<std::collections::HashMap<String, f32>, Error> {
        use std::collections::HashMap;

        let output_labels =
            &self.context.model_config.classifier_config.output_labels;
        let mut probabilities = HashMap::new();

        // Apply sigmoid on CPU to convert linear logits to probabilities
        for (idx, &logit) in logits.iter().enumerate() {
            let prob = 1.0 / (1.0 + (-logit).exp());

            let label = if let Some(labels) = output_labels {
                labels
                    .get(idx)
                    .map(|s| s.clone())
                    .unwrap_or_else(|| format!("class_{}", idx))
            } else {
                format!("class_{}", idx)
            };

            probabilities.insert(label, prob);
        }

        Ok(probabilities)
    }
}
