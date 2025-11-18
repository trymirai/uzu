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
            eprintln!(
                "[DEBUG] classify_tokens - Forward pass completed in {:.2}s",
                forward_duration
            );

            eprintln!(
                "[DEBUG] classify_tokens - Converting to probabilities..."
            );
            let probabilities = self.logits_to_probabilities(&logits)?;

            let stats = ClassificationStats::new(
                forward_duration,
                run_start.elapsed().as_secs_f64(),
            );

            eprintln!(
                "[DEBUG] classify_tokens - Done in {:.2}s",
                run_start.elapsed().as_secs_f64()
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

            eprintln!(
                "[DEBUG] forward_pass - Token IDs (first 10): {:?}",
                &token_ids[..token_ids.len().min(10)]
            );
            eprintln!(
                "[DEBUG] forward_pass - Token positions (first 10): {:?}",
                &token_positions[..token_positions.len().min(10)]
            );

            self.context.embed.encode(
                &mut state,
                &self.context.command_buffer,
                &encoding_params,
            );

            eprintln!(
                "[DEBUG] forward_pass - Encoding embedding normalization..."
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

            eprintln!(
                "[DEBUG] forward_pass - Encoding {} layers...",
                self.context.layers.len()
            );
            for (i, layer) in self.context.layers.iter().enumerate() {
                eprintln!(
                    "[DEBUG] forward_pass - Encoding layer {}/{}...",
                    i + 1,
                    self.context.layers.len()
                );
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

            eprintln!(
                "[DEBUG] forward_pass - Encoding output normalization..."
            );
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
            let logits_linear_array =
                self.apply_prediction_head(&mut state, &mut pooled_output)?;

            // Copy result from GPU to CPU (Array::buffer() will implicitly sync)
            let logits = self.copy_logits_to_cpu(&logits_linear_array)?;

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
        eprintln!(
            "[DEBUG] apply_pooling - batch_size={}, seq_len={}, model_dim={}",
            batch_size, seq_len, model_dim
        );

        let arrays = state.arrays(&[ArrayId::Main]);
        let data_type = {
            use crate::Array;
            Array::data_type(&*arrays[0].borrow())
        };
        let mut main_array = arrays[0].borrow_mut();
        let input_buffer = unsafe { main_array.mtl_buffer() };

        // Use pre-allocated pooling buffer from context (clone to avoid borrow issues)
        let output_buffer = self.context.pooled_buffer.clone();

        self.context.reset_command_buffer();
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
            eprintln!(
                "[DEBUG] apply_pooling - Copying pooled output to trace buffer..."
            );
            let traces_ref = traces_rc.borrow();
            let mut trace_arr = traces_ref.output_pooling.borrow_mut();
            // blit copy pooled_output -> trace_arr
            let dst_buf = unsafe { trace_arr.mtl_buffer().to_owned() };
            drop(trace_arr);
            drop(traces_ref);
            // start a fresh command buffer for blit
            self.context.reset_command_buffer();
            let root2 = self.context.command_buffer.root_command_buffer();
            let blit = root2.new_blit_command_encoder();
            blit.copy_from_buffer(
                &output_buffer,
                0,
                &dst_buf,
                0,
                (batch_size * model_dim * data_type.size_in_bytes()) as u64,
            );
            blit.end_encoding();
            // commit and wait for the blit to complete
            let root_owned2 = root2.to_owned();
            self.context.command_buffer.commit_and_continue();
            root_owned2.wait_until_completed();
            eprintln!(
                "[DEBUG] apply_pooling - Pooled output trace copy completed"
            );
        }

        let pooled_array = unsafe {
            MetalArray::new(
                output_buffer.clone(),
                &[batch_size, model_dim],
                data_type,
            )
        };

        Ok(pooled_array)
    }

    fn apply_prediction_head(
        &mut self,
        state: &mut ClassificationForwardPassState,
        pooled_input: &mut MetalArray,
    ) -> Result<MetalArray, Error> {
        let batch_size = {
            use crate::Array;
            Array::shape(pooled_input)[0]
        };
        let num_labels = self.context.model_config.classifier_config.num_labels;
        let model_dim = self.context.model_config.classifier_config.model_dim;
        let data_type = {
            use crate::Array;
            Array::data_type(pooled_input)
        };
        eprintln!(
            "[DEBUG] apply_prediction_head - batch_size={}, num_labels={}, model_dim={}",
            batch_size, num_labels, model_dim
        );

        // Copy pooled input into ClassifierPredictionHeadPooled buffer
        {
            let pooled_arrays =
                state.arrays(&[ArrayId::ClassifierPredictionHeadPooled]);
            pooled_arrays[0].borrow_mut().copy_from_array(pooled_input);
        }

        let encoding_params = EncodingParameters::new(false, true, false);

        // RUN ALL 4 PREDICTION HEAD OPERATIONS using the single state
        // Pipeline: ClassifierPredictionHeadPooled → Dense → Norm → Logits

        // Step 1: Dense (Pooled → Dense)
        eprintln!(
            "[DEBUG] apply_prediction_head - Step 1: Dense (Pooled → Dense)"
        );
        self.context.prediction_head_dense.encode(
            state,
            &self.context.command_buffer,
            &encoding_params,
        );

        // Step 2: GELU (Dense → Dense, in-place)
        eprintln!(
            "[DEBUG] apply_prediction_head - Step 2: GELU (Dense → Dense)"
        );
        self.context.prediction_head_activation.encode(
            state,
            &self.context.command_buffer,
            &encoding_params,
        );

        // Step 3: Norm (Dense → Norm)
        eprintln!(
            "[DEBUG] apply_prediction_head - Step 3: Norm (Dense → Norm)"
        );
        self.context.prediction_head_norm.encode(
            state,
            &self.context.command_buffer,
            &encoding_params,
        );

        // Step 4: Final linear (Norm → Logits)
        eprintln!(
            "[DEBUG] apply_prediction_head - Step 4: Final linear (Norm → Logits)"
        );
        self.context.prediction_head_final_linear.encode(
            state,
            &self.context.command_buffer,
            &encoding_params,
        );

        self.context.command_buffer.commit_and_continue();

        // Get result from ClassifierPredictionHeadLogits buffer
        let logits_arrays =
            state.arrays(&[ArrayId::ClassifierPredictionHeadLogits]);
        let mut logits_array_ref = logits_arrays[0].borrow_mut();

        eprintln!(
            "[DEBUG] apply_prediction_head - Logits array shape after final_linear: {:?}",
            {
                use crate::Array;
                Array::shape(&*logits_array_ref)
            }
        );

        // Log linear outputs before sigmoid as f32
        let linear_cpu_buffer = {
            use crate::Array;
            Array::buffer(&*logits_array_ref)
        };
        {
            let floats: &[f32] = bytemuck::cast_slice(linear_cpu_buffer);
            let to_show = floats.len().min(8);
            eprintln!(
                "[DEBUG] apply_prediction_head - linear first {} values: {:?}",
                to_show,
                &floats[..to_show]
            );
        }

        let linear_output_buffer =
            unsafe { logits_array_ref.mtl_buffer().to_owned() };
        drop(logits_array_ref);

        // Trace: logits (copy linear outputs to trace buffer)
        if let Some(traces_rc) = state.classifier_traces().cloned() {
            eprintln!(
                "[DEBUG] apply_prediction_head - Copying logits to trace buffer..."
            );
            let traces_ref = traces_rc.borrow();
            let mut trace_logits = traces_ref.logits.borrow_mut();
            let dst_trace_buf = unsafe { trace_logits.mtl_buffer().to_owned() };
            drop(trace_logits);
            drop(traces_ref);

            let copy_size_bytes =
                (batch_size * num_labels * data_type.size_in_bytes()) as u64;

            // start a fresh command buffer for blit
            self.context.reset_command_buffer();
            let root2 = self.context.command_buffer.root_command_buffer();
            let blit2 = root2.new_blit_command_encoder();
            blit2.copy_from_buffer(
                &linear_output_buffer,
                0,
                &dst_trace_buf,
                0,
                copy_size_bytes,
            );
            blit2.end_encoding();
            let root_owned2 = root2.to_owned();
            self.context.command_buffer.commit_and_continue();
            root_owned2.wait_until_completed();
            eprintln!(
                "[DEBUG] apply_prediction_head - Logits trace copy completed"
            );
        }

        // Return linear logits array [batch_size, num_labels]
        let output_array = unsafe {
            MetalArray::new(
                linear_output_buffer,
                &[batch_size, num_labels],
                data_type,
            )
        };

        Ok(output_array)
    }

    fn copy_logits_to_cpu(
        &self,
        logits_array: &MetalArray,
    ) -> Result<Vec<f32>, Error> {
        use crate::Array;
        let num_labels = self.context.model_config.classifier_config.num_labels;
        let buffer = Array::buffer(logits_array);

        match Array::data_type(logits_array) {
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
