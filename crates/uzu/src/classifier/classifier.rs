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
            ArrayId, ForwardPassState, ForwardPassStateTrait,
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
        eprintln!("[DEBUG] Classifier::new - Creating context...");
        let context = ClassifierContext::new(model_path)?;
        eprintln!("[DEBUG] Classifier::new - Context created");
        Ok(Self {
            context,
        })
    }

    pub fn classify_tokens(
        &mut self,
        token_ids: Vec<u64>,
        token_positions: Vec<usize>,
    ) -> Result<ClassificationOutput, Error> {
        eprintln!("[DEBUG] classify_tokens - Start");
        let run_start = Instant::now();

        autoreleasepool(|_pool| {
            let forward_start = Instant::now();
            eprintln!("[DEBUG] classify_tokens - Running forward pass...");

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
            eprintln!("[DEBUG] forward_pass - Creating forward pass state...");
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
            eprintln!("[DEBUG] forward_pass - State created");

            let encoding_params = EncodingParameters::new(false, true, false);

            // Reset command buffer for this forward pass
            eprintln!("[DEBUG] forward_pass - Resetting command buffer...");
            self.context.reset_command_buffer();

            eprintln!(
                "[DEBUG] forward_pass - Token IDs (first 10): {:?}",
                &token_ids[..token_ids.len().min(10)]
            );
            eprintln!(
                "[DEBUG] forward_pass - Token positions (first 10): {:?}",
                &token_positions[..token_positions.len().min(10)]
            );

            eprintln!("[DEBUG] forward_pass - Encoding embeddings...");
            self.context.embed.encode(
                &mut state,
                &self.context.command_buffer,
                &encoding_params,
            );

            // Commit embeddings to see intermediate result
            eprintln!("[DEBUG] forward_pass - Committing embeddings...");
            let embed_root =
                self.context.command_buffer.root_command_buffer().to_owned();
            self.context.command_buffer.commit_and_continue();
            embed_root.wait_until_completed();

            // Debug: Check embeddings output
            eprintln!("[DEBUG] forward_pass - Checking embeddings output...");
            {
                use crate::Array;
                let main_arrays = state.arrays(&[ArrayId::Main]);
                let main_array = main_arrays[0].borrow();
                let buffer = Array::buffer(&*main_array);
                eprintln!(
                    "[DEBUG] forward_pass - embeddings Main array shape: {:?}, buffer len: {}",
                    Array::shape(&*main_array),
                    buffer.len()
                );
                eprintln!(
                    "[DEBUG] forward_pass - first 20 bytes after embeddings: {:?}",
                    &buffer[..buffer.len().min(20)]
                );
            }

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

            // Commit and wait for GPU to complete embedding + transformer + output norm
            eprintln!("[DEBUG] forward_pass - Committing command buffer...");
            let root =
                self.context.command_buffer.root_command_buffer().to_owned();
            self.context.command_buffer.commit_and_continue();
            eprintln!(
                "[DEBUG] forward_pass - Waiting for GPU to complete transformer..."
            );
            root.wait_until_completed();
            eprintln!("[DEBUG] forward_pass - Transformer completed");

            // At this point, the transformer output is in the Main array [batch, seq_len, model_dim]

            // Debug: Check transformer output before pooling
            eprintln!("[DEBUG] forward_pass - Checking transformer output...");
            {
                use crate::Array;
                let main_arrays = state.arrays(&[ArrayId::Main]);
                let main_array = main_arrays[0].borrow();
                let buffer = Array::buffer(&*main_array);
                eprintln!(
                    "[DEBUG] forward_pass - transformer Main array shape: {:?}, buffer len: {}",
                    Array::shape(&*main_array),
                    buffer.len()
                );
                eprintln!(
                    "[DEBUG] forward_pass - first 20 bytes of transformer output: {:?}",
                    &buffer[..buffer.len().min(20)]
                );
            }

            // Apply pooling: [batch, seq_len, model_dim] → [batch, model_dim]
            eprintln!("[DEBUG] forward_pass - Applying pooling...");
            let pooled_output = self.apply_pooling(&mut state)?;
            eprintln!("[DEBUG] forward_pass - Pooling completed");

            // Apply prediction head: [batch, model_dim] → [batch, num_labels]
            eprintln!("[DEBUG] forward_pass - Applying prediction head...");
            let logits_linear_array =
                self.apply_prediction_head(&mut state, &pooled_output)?;
            eprintln!("[DEBUG] forward_pass - Prediction head completed");

            // Copy result from GPU to CPU
            eprintln!("[DEBUG] forward_pass - Copying logits to CPU...");
            let logits = self.copy_logits_to_cpu(&logits_linear_array)?;
            eprintln!("[DEBUG] forward_pass - All done");

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
        eprintln!("[DEBUG] apply_pooling - Start");
        let batch_size = 1;
        let seq_len = state.aux_buffers_suffix_length();
        let model_dim = self.context.model_config.classifier_config.model_dim;
        eprintln!(
            "[DEBUG] apply_pooling - batch_size={}, seq_len={}, model_dim={}",
            batch_size, seq_len, model_dim
        );

        eprintln!("[DEBUG] apply_pooling - Getting arrays...");
        let arrays = state.arrays(&[ArrayId::Main]);
        let data_type = {
            use crate::Array;
            Array::data_type(&*arrays[0].borrow())
        };
        let mut main_array = arrays[0].borrow_mut();
        let input_buffer = unsafe { main_array.mtl_buffer() };

        // Use pre-allocated pooling buffer from context (clone to avoid borrow issues)
        let output_buffer = self.context.pooled_buffer.clone();

        eprintln!("[DEBUG] apply_pooling - Encoding pooling operation...");
        self.context.reset_command_buffer();
        let root = self.context.command_buffer.root_command_buffer();
        let encoder = root.new_compute_command_encoder();

        match self.context.model_config.classifier_config.classifier_pooling {
            PoolingType::Cls => {
                eprintln!("[DEBUG] apply_pooling - Using CLS pooling");
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
                eprintln!("[DEBUG] apply_pooling - Using Mean pooling");
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

        eprintln!("[DEBUG] apply_pooling - Ending encoding...");
        encoder.end_encoding();
        drop(main_array);

        eprintln!("[DEBUG] apply_pooling - Committing...");
        let root_owned = root.to_owned();
        self.context.command_buffer.commit_and_continue();
        eprintln!("[DEBUG] apply_pooling - Waiting for completion...");
        root_owned.wait_until_completed();
        eprintln!("[DEBUG] apply_pooling - Pooling GPU work completed");

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

        // Debug: print a few pooled feature values as f32
        {
            use crate::Array;
            let bytes = Array::buffer(&pooled_array);
            let floats: &[f32] = bytemuck::cast_slice(bytes);
            let to_show = floats.len().min(8);
            eprintln!(
                "[DEBUG] apply_pooling - pooled first {} values: {:?}",
                to_show,
                &floats[..to_show]
            );
        }

        eprintln!("[DEBUG] apply_pooling - Done");
        Ok(pooled_array)
    }

    fn apply_prediction_head(
        &mut self,
        state: &mut ClassificationForwardPassState,
        pooled_input: &MetalArray,
    ) -> Result<MetalArray, Error> {
        eprintln!("[DEBUG] apply_prediction_head - Start");
        // Prediction head: dense → GELU → norm → final_linear
        // All operations on Metal

        let batch_size = {
            use crate::Array;
            Array::shape(pooled_input)[0]
        };
        let num_labels = self.context.model_config.classifier_config.num_labels;
        let data_type = {
            use crate::Array;
            Array::data_type(pooled_input)
        };
        eprintln!(
            "[DEBUG] apply_prediction_head - batch_size={}, num_labels={}",
            batch_size, num_labels
        );

        // Create a minimal ForwardPassState for prediction head
        // We need at least 1 dummy token to allocate buffers, even though we won't use sequence dimension
        eprintln!(
            "[DEBUG] apply_prediction_head - Creating prediction state..."
        );
        let decoder_config =
            self.context.model_config.classifier_config.to_decoder_config();
        let dummy_tokens = vec![0u64; 1]; // 1 dummy token to allocate buffers
        let dummy_positions = vec![0usize; 1];
        let mut pred_state = ForwardPassState::new(
            self.context.mtl_context.clone(),
            &decoder_config,
            &self.context.model_shape,
            &self.context.scratch_buffers,
            self.context.kv_cache.clone(),
            self.context.shared_buffers.clone(),
            &dummy_tokens,
            &dummy_positions,
            false,
            None,
        );

        // Copy pooled input to Main array in pred_state
        // The pooled input is [batch=1, model_dim] and we need to write it as [batch=1, seq=1, model_dim]
        eprintln!(
            "[DEBUG] apply_prediction_head - Copying pooled input to state..."
        );
        let main_arrays = pred_state.arrays(&[ArrayId::Main]);
        {
            use crate::Array;
            let pooled_buffer = Array::buffer(pooled_input);
            let pooled_shape = Array::shape(pooled_input);

            let mut main_array = main_arrays[0].borrow_mut();
            let main_shape = Array::shape(&*main_array).to_vec();
            let buffer = Array::buffer_mut(&mut *main_array);

            eprintln!(
                "[DEBUG] apply_prediction_head - main_array shape: {:?}, pooled shape: {:?}",
                main_shape, pooled_shape
            );
            eprintln!(
                "[DEBUG] apply_prediction_head - main_array buffer len: {}, pooled buffer len: {}",
                buffer.len(),
                pooled_buffer.len()
            );
            if buffer.len() >= pooled_buffer.len() {
                buffer[..pooled_buffer.len()].copy_from_slice(pooled_buffer);
                // Log first few values
                eprintln!(
                    "[DEBUG] apply_prediction_head - first 5 pooled values (as bytes): {:?}",
                    &pooled_buffer[..pooled_buffer.len().min(20)]
                );
            } else {
                return Err(Error::PrefillFailed);
            }
        }

        // Reset command buffer
        eprintln!(
            "[DEBUG] apply_prediction_head - Resetting command buffer..."
        );
        self.context.reset_command_buffer();
        let encoding_params = EncodingParameters::new(false, true, false);

        // Run prediction head layers in sequence
        // 1. Dense layer
        eprintln!("[DEBUG] apply_prediction_head - Encoding dense layer...");
        self.context.prediction_head.dense.encode(
            &mut pred_state,
            &self.context.command_buffer,
            &encoding_params,
        );

        // 2. Activation (GELU)
        eprintln!(
            "[DEBUG] apply_prediction_head - Encoding activation (GELU)..."
        );
        self.context.prediction_head.activation.encode(
            &mut pred_state,
            &self.context.command_buffer,
            &encoding_params,
        );

        // 3. Normalization
        eprintln!("[DEBUG] apply_prediction_head - Encoding normalization...");
        self.context.prediction_head.norm.encode(
            &mut pred_state,
            &self.context.command_buffer,
            &encoding_params,
        );

        // 4. Final linear layer
        eprintln!(
            "[DEBUG] apply_prediction_head - Encoding final linear layer..."
        );
        self.context.prediction_head.final_linear.encode(
            &mut pred_state,
            &self.context.command_buffer,
            &encoding_params,
        );

        // Commit and wait for prediction head to complete
        eprintln!(
            "[DEBUG] apply_prediction_head - Committing prediction head..."
        );
        let root = self.context.command_buffer.root_command_buffer().to_owned();
        self.context.command_buffer.commit_and_continue();
        eprintln!(
            "[DEBUG] apply_prediction_head - Waiting for prediction head GPU work..."
        );
        root.wait_until_completed();
        eprintln!(
            "[DEBUG] apply_prediction_head - Prediction head GPU work completed"
        );

        // Now apply sigmoid to convert linear outputs to probabilities.
        // IMPORTANT: Prediction head writes to Main buffer with shape [batch, num_labels].
        eprintln!(
            "[DEBUG] apply_prediction_head - Getting linear output from Main..."
        );
        let main_after_arrays = pred_state.arrays(&[ArrayId::Main]);
        let mut main_after = main_after_arrays[0].borrow_mut();
        eprintln!(
            "[DEBUG] apply_prediction_head - Main array shape after final_linear: {:?}",
            {
                use crate::Array;
                Array::shape(&*main_after)
            }
        );
        // Log linear outputs before sigmoid as f32
        let linear_cpu_buffer = {
            use crate::Array;
            Array::buffer(&*main_after)
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
            unsafe { main_after.mtl_buffer().to_owned() };

        // Copy linear outputs (first num_labels values) into a dedicated logits buffer
        eprintln!(
            "[DEBUG] apply_prediction_head - Copying linear outputs to final logits buffer..."
        );
        let output_size = batch_size * num_labels;
        let copy_size_bytes = (output_size * data_type.size_in_bytes()) as u64;
        let dst_buf = self.context.final_logits_buffer.clone();

        // Reset command buffer for blit copy
        eprintln!("[DEBUG] apply_prediction_head - Encoding blit copy...");
        self.context.reset_command_buffer();
        let root = self.context.command_buffer.root_command_buffer();
        let blit = root.new_blit_command_encoder();
        blit.copy_from_buffer(
            &linear_output_buffer,
            0,
            &dst_buf,
            0,
            copy_size_bytes,
        );
        blit.end_encoding();
        drop(main_after);

        // Commit and wait for blit to complete
        eprintln!("[DEBUG] apply_prediction_head - Committing blit copy...");
        let root_owned = root.to_owned();
        self.context.command_buffer.commit_and_continue();
        eprintln!("[DEBUG] apply_prediction_head - Waiting for GPU blit...");
        root_owned.wait_until_completed();
        eprintln!("[DEBUG] apply_prediction_head - Blit copy completed");

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
            // start a fresh command buffer for blit
            self.context.reset_command_buffer();
            let root2 = self.context.command_buffer.root_command_buffer();
            let blit2 = root2.new_blit_command_encoder();
            blit2.copy_from_buffer(
                &dst_buf,
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
                dst_buf.clone(),
                &[batch_size, num_labels],
                data_type,
            )
        };

        eprintln!("[DEBUG] apply_prediction_head - Done");
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

    pub fn reset(&mut self) {
        // no-op
    }
}
