use std::{path::Path, time::Instant};

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

            let logits = self.forward_pass(&token_ids, &token_positions)?;

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

    fn forward_pass(
        &mut self,
        token_ids: &[u64],
        token_positions: &[usize],
    ) -> Result<Vec<f32>, Error> {
        autoreleasepool(|_| {
            eprintln!("[DEBUG] forward_pass - Creating forward pass state...");
            // Create ClassificationForwardPassState with bidirectional attention
            // No KV cache, no vocabulary-sized buffers - designed for classification
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
            }

            eprintln!(
                "[DEBUG] forward_pass - Encoding output normalization..."
            );
            self.context.output_norm.encode(
                &mut state,
                &self.context.command_buffer,
                &encoding_params,
            );

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
            let logits_with_sigmoid =
                self.apply_prediction_head(&pooled_output)?;
            eprintln!("[DEBUG] forward_pass - Prediction head completed");

            // Copy result from GPU to CPU
            eprintln!("[DEBUG] forward_pass - Copying logits to CPU...");
            let logits = self.copy_logits_to_cpu(&logits_with_sigmoid)?;
            eprintln!("[DEBUG] forward_pass - All done");

            Ok(logits)
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

        eprintln!("[DEBUG] apply_pooling - Creating output buffer...");
        let output_size = batch_size * model_dim;
        let output_buffer = self.context.mtl_context.device.new_buffer(
            (output_size * data_type.size_in_bytes()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

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

        let pooled_array = unsafe {
            MetalArray::new(output_buffer, &[batch_size, model_dim], data_type)
        };

        eprintln!("[DEBUG] apply_pooling - Done");
        Ok(pooled_array)
    }

    fn apply_prediction_head(
        &mut self,
        pooled_input: &MetalArray,
    ) -> Result<MetalArray, Error> {
        eprintln!("[DEBUG] apply_prediction_head - Start");
        // Prediction head: dense → GELU → norm → final_linear → sigmoid
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

        // Now apply sigmoid to convert logits to probabilities
        // Read from Logits array which has the correct shape [1, num_labels]
        eprintln!("[DEBUG] apply_prediction_head - Getting logits array...");
        let logits_arrays = pred_state.arrays(&[ArrayId::Logits]);
        let mut logits_array = logits_arrays[0].borrow_mut();
        eprintln!(
            "[DEBUG] apply_prediction_head - logits array shape: {:?}",
            {
                use crate::Array;
                Array::shape(&*logits_array)
            }
        );
        // Log logits values before sigmoid
        let logits_cpu_buffer = {
            use crate::Array;
            Array::buffer(&*logits_array)
        };
        eprintln!(
            "[DEBUG] apply_prediction_head - first 20 bytes of logits: {:?}",
            &logits_cpu_buffer[..logits_cpu_buffer.len().min(20)]
        );
        let logits_buffer = unsafe { logits_array.mtl_buffer() };

        // Allocate output buffer for sigmoid result
        eprintln!(
            "[DEBUG] apply_prediction_head - Creating sigmoid output buffer..."
        );
        let output_size = batch_size * num_labels;
        let sigmoid_output_buffer = self.context.mtl_context.device.new_buffer(
            (output_size * data_type.size_in_bytes()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Reset command buffer for sigmoid
        eprintln!("[DEBUG] apply_prediction_head - Encoding sigmoid...");
        self.context.reset_command_buffer();
        let root = self.context.command_buffer.root_command_buffer();
        let encoder = root.new_compute_command_encoder();

        // Apply sigmoid
        self.context
            .sigmoid_kernel
            .encode(
                encoder,
                logits_buffer,
                &sigmoid_output_buffer,
                output_size as i32,
            )
            .map_err(|_| Error::GenerateFailed)?;

        eprintln!("[DEBUG] apply_prediction_head - Ending sigmoid encoding...");
        encoder.end_encoding();
        drop(logits_array);

        // Commit and wait for sigmoid to complete
        eprintln!("[DEBUG] apply_prediction_head - Committing sigmoid...");
        let root_owned = root.to_owned();
        self.context.command_buffer.commit_and_continue();
        eprintln!(
            "[DEBUG] apply_prediction_head - Waiting for sigmoid GPU work..."
        );
        root_owned.wait_until_completed();
        eprintln!("[DEBUG] apply_prediction_head - Sigmoid GPU work completed");

        // Return sigmoid output (probabilities)
        let output_array = unsafe {
            MetalArray::new(
                sigmoid_output_buffer,
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
