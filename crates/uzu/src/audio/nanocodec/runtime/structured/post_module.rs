use super::*;

impl StructuredAudioCodecGraph {
    pub(super) fn apply_convnext_ncs_enqueued(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
        command_buffer: &mut MetalCommandBuffer,
        input: &Array<Metal>,
        layer: &StructuredAudioConvNeXt,
        lengths: &[i32],
        lengths_array: &Array<Metal>,
        batch_size: usize,
        channels: usize,
        seq_len: usize,
    ) -> AudioResult<Array<Metal>> {
        let residual = input.clone();
        let x = causal_conv1d_grouped_enqueue(
            context,
            command_buffer,
            input,
            &layer.depthwise_conv,
            SequenceLayout::Ncs,
            lengths,
            lengths_array,
            batch_size,
            seq_len,
        )?;
        let x = norm_ncs_enqueue(
            context,
            command_buffer,
            &x,
            &layer.norm,
            lengths,
            lengths_array,
            batch_size,
            channels,
            seq_len,
        )?;
        let x = conv1d_pointwise_ncs_enqueue(
            context,
            command_buffer,
            &x,
            &layer.pwconv1,
            lengths,
            lengths_array,
            batch_size,
            seq_len,
        )?;
        let x = gelu_enqueue(context, command_buffer, &x)?;
        let x = conv1d_pointwise_ncs_enqueue(
            context,
            command_buffer,
            &x,
            &layer.pwconv2,
            lengths,
            lengths_array,
            batch_size,
            seq_len,
        )?;
        add_enqueue(context, command_buffer, &x, &residual)
    }

    pub(super) fn build_post_module_runtime(
        &self,
        context: Rc<<Metal as Backend>::Context>,
        required_sequence_length: usize,
    ) -> AudioResult<StructuredAudioPostModuleRuntime> {
        let inner_model_config = InnerModelConfig {
            embedding_config: EmbeddingConfig::Untied {
                common: EmbeddingConfigCommon {
                    input_scale: None,
                    logit_soft_cap: None,
                },
                precision: self.vocoder_data_type.into(),
            },
            transformer_config: self.config.quantizer_config.post_module_config.clone(),
            vocab_size: 1,
        };
        let decoder_config =
            Rc::new(inner_model_config.to_decoder_config().map_err(|_| {
                AudioError::Runtime("failed to build structured audio post_module decoder config".to_string())
            })?);
        let model_shape = ModelShape::from_decoder_config(&decoder_config);

        let weights_file = File::open(self.weights_path.as_str()).map_err(|err| {
            AudioError::Runtime(format!("failed to open post_module weights '{}': {err}", self.weights_path))
        })?;
        let loader = ParameterLoader::new(&weights_file, context.as_ref()).map_err(|err| {
            AudioError::Runtime(format!("failed to load post_module weights '{}': {err}", self.weights_path))
        })?;
        let root_loader_view = loader.tree();
        let transformer_subtree_name = "audio_decoder.quantizer.post_module";
        root_loader_view
            .subtree(transformer_subtree_name)
            .map_err(|err| AudioError::Runtime(format!("missing structured audio post_module subtree: {err}")))?;

        let max_sequence_length = decoder_config.context_length.max(required_sequence_length.max(1));
        let shared_buffers = Rc::new(RefCell::new(SharedBuffers::new(context.as_ref(), &decoder_config, &model_shape)));
        {
            let transformer_tree = root_loader_view
                .subtree(transformer_subtree_name)
                .map_err(|err| AudioError::Runtime(format!("missing structured audio post_module subtree: {err}")))?;
            let mut shared_buffers = shared_buffers.borrow_mut();
            if let Some(global_rope) = &mut shared_buffers.global_rope {
                global_rope.update_data(&transformer_tree, "global_rope");
            }
            if let Some(local_rope) = &mut shared_buffers.local_rope {
                local_rope.update_data(&transformer_tree, "local_rope");
            }
        }
        let scratch_buffers = ScratchBuffers::new(
            context.as_ref(),
            &decoder_config,
            &model_shape,
            max_sequence_length,
            max_sequence_length,
        );
        let (layers, output_norm) = Decoder::build_transformer_layers_and_norm(
            context.clone(),
            decoder_config,
            &root_loader_view,
            transformer_subtree_name,
        );

        Ok(StructuredAudioPostModuleRuntime {
            context,
            model_shape,
            scratch_buffers,
            shared_buffers,
            layers,
            output_norm,
            max_sequence_length,
        })
    }

    pub(super) fn post_module_runtime_on_context(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
        required_sequence_length: usize,
    ) -> AudioResult<Rc<StructuredAudioPostModuleRuntime>> {
        let cache_key = format!("{}::{}", self.weights_path, Rc::as_ptr(context) as usize);
        FISHAUDIO_POST_MODULE_RUNTIME_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if let Some(runtime) = cache.get(cache_key.as_str()) {
                if runtime.max_sequence_length >= required_sequence_length.max(1) {
                    return Ok(runtime.clone());
                }
            }

            let runtime = Rc::new(self.build_post_module_runtime(context.clone(), required_sequence_length)?);
            cache.insert(cache_key, runtime.clone());
            Ok(runtime)
        })
    }

    pub(super) fn decode_context(&self) -> AudioResult<Rc<<Metal as Backend>::Context>> {
        FISHAUDIO_DECODE_CONTEXT_CACHE.with(|cache| {
            if let Some(existing) = cache.borrow().get(&self.weights_path).cloned() {
                return Ok(existing);
            }
            let created = <Metal as Backend>::Context::new()
                .map_err(|err| AudioError::Runtime(format!("failed to create structured audio decode context: {err}")))?;
            cache.borrow_mut().insert(self.weights_path.clone(), created.clone());
            Ok(created)
        })
    }

    pub(super) fn build_quantizer_gpu_resources(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
    ) -> AudioResult<FishAudioQuantizerResources> {
        if self.semantic_codebook_size == 0 || self.codebook_size == 0 {
            return Err(AudioError::InvalidTokenCardinality);
        }
        let data_type = self.vocoder_data_type;
        let kernel =
            <<Metal as Backend>::Kernels as Kernels>::AudioQuantizerDecodeKernel::new(context.as_ref(), data_type)
                .map_err(|err| AudioError::Runtime(format!("failed to initialize quantizer decode kernel: {err}")))?;
        let weights_file = File::open(self.weights_path.as_str()).map_err(|err| {
            AudioError::Runtime(format!(
                "failed to open structured audio weights '{}': {err}",
                self.weights_path
            ))
        })?;
        let loader = ParameterLoader::new(&weights_file, context.as_ref()).map_err(|err| {
            AudioError::Runtime(format!(
                "failed to load structured audio weights '{}': {err}",
                self.weights_path
            ))
        })?;
        let root = loader.tree();
        let audio_decoder_tree = root.subtree("audio_decoder")?;
        let quantizer_tree = audio_decoder_tree.subtree("quantizer")?;
        let semantic_tree = quantizer_tree.subtree("semantic_quantizer")?.subtree("quantizers")?.subtree("0")?;
        let semantic_codebook = read_float_matrix_exact(
            &semantic_tree.subtree("codebook")?,
            "weights",
            self.semantic_codebook_size,
            self.config.codebook_dim,
            data_type,
        )?;
        let codebook_dim = semantic_codebook.shape()[1];
        let semantic_out_proj = read_float_matrix_exact(
            &semantic_tree.subtree("out_proj")?,
            "weights",
            self.input_dim,
            codebook_dim,
            data_type,
        )?;
        let semantic_out_bias =
            read_float_vector_exact(&semantic_tree.subtree("out_proj")?, "biases", self.input_dim, data_type)?;

        let residual_quantizers = self.config.n_codebooks;
        let residual_count_for_shape = residual_quantizers.max(1);
        let residual_codebook_rows_for_shape = self.codebook_size.max(1);
        let residual_codebooks = context.create_array(
            &[residual_count_for_shape, residual_codebook_rows_for_shape, codebook_dim],
            data_type,
            "structured_audio_quantizer_residual_codebooks",
        );
        let residual_out_proj = context.create_array(
            &[residual_count_for_shape, self.input_dim, codebook_dim],
            data_type,
            "structured_audio_quantizer_residual_out_proj",
        );
        let residual_out_bias = context.create_array(
            &[residual_count_for_shape, self.input_dim],
            data_type,
            "structured_audio_quantizer_residual_out_bias",
        );

        let residual_root = quantizer_tree.subtree("quantizer")?.subtree("quantizers")?;
        for index in 0..residual_quantizers {
            let quantizer_tree = residual_root.subtree(&index.to_string())?;
            let codebook = read_float_matrix_exact(
                &quantizer_tree.subtree("codebook")?,
                "weights",
                self.codebook_size,
                codebook_dim,
                data_type,
            )?;
            let out_proj = read_float_matrix_exact(
                &quantizer_tree.subtree("out_proj")?,
                "weights",
                self.input_dim,
                codebook_dim,
                data_type,
            )?;
            let out_bias =
                read_float_vector_exact(&quantizer_tree.subtree("out_proj")?, "biases", self.input_dim, data_type)?;

            let mut dst_codebook = outer_axis_view(&residual_codebooks, index, &[self.codebook_size, codebook_dim])?;
            dst_codebook.copy_from_array(&codebook);
            let mut dst_out_proj = outer_axis_view(&residual_out_proj, index, &[self.input_dim, codebook_dim])?;
            dst_out_proj.copy_from_array(&out_proj);
            let mut dst_out_bias = outer_axis_view(&residual_out_bias, index, &[self.input_dim])?;
            dst_out_bias.copy_from_array(&out_bias);
        }

        Ok(FishAudioQuantizerResources {
            data_type,
            codebook_dim,
            residual_quantizers,
            semantic_cardinality: self.semantic_codebook_size,
            residual_cardinality: self.codebook_size,
            kernel,
            semantic_codebook,
            semantic_out_proj,
            semantic_out_bias,
            residual_codebooks,
            residual_out_proj,
            residual_out_bias,
        })
    }

    pub(super) fn quantizer_gpu_resources(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
    ) -> AudioResult<Rc<FishAudioQuantizerResources>> {
        let key = (Rc::as_ptr(context) as usize).wrapping_mul(31) ^ usize::from(structured_audio_dtype_key(self.vocoder_data_type));
        FISHAUDIO_QUANTIZER_RESOURCES_CACHE.with(|cache| {
            if let Some(existing) = cache.borrow().get(&key) {
                return Ok(existing.clone());
            }
            let resources = Rc::new(self.build_quantizer_gpu_resources(context)?);
            cache.borrow_mut().insert(key, resources.clone());
            Ok(resources)
        })
    }

    pub(super) fn build_vocoder_gpu_graph(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
    ) -> AudioResult<StructuredAudioDecoderGraph> {
        let weights_file = File::open(self.weights_path.as_str()).map_err(|err| {
            AudioError::Runtime(format!(
                "failed to open structured audio weights '{}': {err}",
                self.weights_path
            ))
        })?;
        let loader = ParameterLoader::new(&weights_file, context.as_ref()).map_err(|err| {
            AudioError::Runtime(format!(
                "failed to load structured audio weights '{}': {err}",
                self.weights_path
            ))
        })?;
        let root = loader.tree();
        build_vocoder_gpu_graph_from_tree(context, &root, &self.config, self.vocoder_data_type)
    }

    pub(super) fn vocoder_gpu_graph(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
    ) -> AudioResult<Rc<StructuredAudioDecoderGraph>> {
        let key = (Rc::as_ptr(context) as usize).wrapping_mul(31) ^ usize::from(structured_audio_dtype_key(self.vocoder_data_type));
        FISHAUDIO_VOCODER_GRAPH_CACHE.with(|cache| {
            if let Some(existing) = cache.borrow().get(&key) {
                return Ok(existing.clone());
            }
            let graph = Rc::new(self.build_vocoder_gpu_graph(context)?);
            cache.borrow_mut().insert(key, graph.clone());
            Ok(graph)
        })
    }

    pub(super) fn encode_post_module_layers(
        runtime: &StructuredAudioPostModuleRuntime,
        state: &mut ForwardPassState<Metal>,
        command_buffer: &mut MetalCommandBuffer,
    ) -> AudioResult<()> {
        let encoding_parameters = EncodingParameters::new();
        for layer in runtime.layers.iter() {
            layer
                .encode(state, &encoding_parameters, command_buffer)
                .map_err(|err| AudioError::Runtime(format!("post_module layer encode failed: {err}")))?;
        }
        runtime
            .output_norm
            .encode(state, command_buffer)
            .map_err(|err| AudioError::Runtime(format!("post_module output norm encode failed: {err}")))?;
        Ok(())
    }

    pub(super) fn apply_post_module_gpu_on_array_single_batch(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
        latent_nsc: &Array<Metal>,
        frames: usize,
        profile: &mut Option<AudioDecodeProfile>,
    ) -> AudioResult<Array<Metal>> {
        if frames == 0 {
            return Ok(latent_nsc.clone());
        }

        let expected_elements = checked_product(&[frames, self.input_dim])?;
        if latent_nsc.num_elements() != expected_elements {
            return Err(AudioError::InvalidTokenShape {
                expected_tokens: expected_elements,
                actual_tokens: latent_nsc.num_elements(),
            });
        }

        let runtime = self.post_module_runtime_on_context(context, frames.max(1))?;
        let token_ids = vec![0_u64; frames];
        let token_positions = (0..frames).collect::<Vec<_>>();
        let mut state = ForwardPassState::new_classifier(
            runtime.context.clone(),
            &runtime.model_shape,
            &runtime.scratch_buffers,
            runtime.shared_buffers.clone(),
            &token_ids,
            &token_positions,
            false,
            1,
        );

        let main = state.arrays(&[ArrayId::Main])[0].clone();
        let main_output = {
            let main_ref = main.borrow();
            if main_ref.shape() != [frames, self.input_dim] {
                return Err(AudioError::Runtime(format!(
                    "post_module main shape mismatch: expected [{frames}, {}], got {:?}",
                    self.input_dim,
                    main_ref.shape()
                )));
            }
            if main_ref.data_type() != latent_nsc.data_type() {
                return Err(AudioError::Runtime(format!(
                    "post_module dtype mismatch: main={:?}, latent={:?}",
                    main_ref.data_type(),
                    latent_nsc.data_type()
                )));
            }
            main_ref.clone()
        };

        let copy_bytes = latent_nsc
            .num_elements()
            .checked_mul(latent_nsc.data_type().size_in_bytes())
            .ok_or(AudioError::Runtime("post_module copy size overflow".to_string()))?;
        let encode_start = profile.is_some().then(Instant::now);
        let mut command_buffer = runtime
            .context
            .create_command_buffer()
            .map_err(|err| AudioError::Runtime(format!("failed to create post_module command buffer: {err}")))?
            .start_encoding();
        command_buffer.with_copy_encoder(|copy_encoder| {
            let latent_buffer = latent_nsc.buffer();
            let main_output_buffer = main_output.buffer();
            let latent_buffer = latent_buffer.borrow();
            let main_output_buffer = main_output_buffer.borrow();
            copy_encoder.encode_copy(&latent_buffer, &main_output_buffer, copy_bytes);
        });
        Self::encode_post_module_layers(&runtime, &mut state, &mut command_buffer)?;
        let cpu_encode_ms = encode_start.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
        let command_buffer = command_buffer.end_encoding().submit();
        let wait_start = profile.is_some().then(Instant::now);
        let command_buffer = command_buffer
            .wait_until_completed()
            .map_err(|err| AudioError::Runtime(format!("failed to wait for post_module command buffer: {err}")))?;
        let cpu_wait_ms = wait_start.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
        push_audio_command_buffer_profile(profile, "post_module", &command_buffer, cpu_encode_ms, cpu_wait_ms, None);

        Ok(main_output)
    }

    pub(super) fn apply_post_module_gpu_on_array(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
        latent_nsc: &Array<Metal>,
        lengths: &[usize],
        batch_size: usize,
        frames: usize,
        profile: &mut Option<AudioDecodeProfile>,
    ) -> AudioResult<Array<Metal>> {
        if lengths.len() != batch_size {
            return Err(AudioError::InvalidTokenLengths {
                expected_lengths: batch_size,
                actual_lengths: lengths.len(),
            });
        }
        let expected_elements = checked_product(&[batch_size, frames, self.input_dim])?;
        if latent_nsc.num_elements() != expected_elements {
            return Err(AudioError::InvalidTokenShape {
                expected_tokens: expected_elements,
                actual_tokens: latent_nsc.num_elements(),
            });
        }
        if batch_size == 1 && lengths.first().copied() == Some(frames) {
            return self.apply_post_module_gpu_on_array_single_batch(context, latent_nsc, frames, profile);
        }

        let output = context.create_array(
            &[batch_size, frames, self.input_dim],
            latent_nsc.data_type(),
            "structured_audio_post_module_output_nsc",
        );
        let mut batch_indices_by_length = BTreeMap::<usize, Vec<usize>>::new();
        for (batch_index, &active_len) in lengths.iter().enumerate() {
            if active_len == 0 {
                continue;
            }
            if active_len > frames {
                return Err(AudioError::InvalidTokenLengthValue {
                    length: active_len,
                    frames,
                });
            }
            batch_indices_by_length.entry(active_len).or_default().push(batch_index);
        }

        if batch_indices_by_length.is_empty() {
            let mut output = output;
            output.copy_from_array(latent_nsc);
            return Ok(output);
        }

        let full_copy_bytes = latent_nsc.size();
        let mut copied_output_prefix = false;
        for (active_len, batch_indices) in batch_indices_by_length {
            let runtime = self.post_module_runtime_on_context(context, active_len.max(1))?;
            let token_ids = vec![0_u64; active_len];
            let token_positions = (0..active_len).collect::<Vec<_>>();
            let mut state = ForwardPassState::new_classifier(
                runtime.context.clone(),
                &runtime.model_shape,
                &runtime.scratch_buffers,
                runtime.shared_buffers.clone(),
                &token_ids,
                &token_positions,
                false,
                1,
            );

            let main = state.arrays(&[ArrayId::Main])[0].clone();
            let main_output = {
                let main_ref = main.borrow();
                if main_ref.shape() != [active_len, self.input_dim] {
                    return Err(AudioError::Runtime(format!(
                        "post_module main shape mismatch: expected [{active_len}, {}], got {:?}",
                        self.input_dim,
                        main_ref.shape()
                    )));
                }
                if main_ref.data_type() != latent_nsc.data_type() {
                    return Err(AudioError::Runtime(format!(
                        "post_module dtype mismatch: main={:?}, latent={:?}",
                        main_ref.data_type(),
                        latent_nsc.data_type()
                    )));
                }
                main_ref.clone()
            };

            for &batch_index in &batch_indices {
                let encode_start = profile.is_some().then(Instant::now);
                let mut command_buffer = runtime
                    .context
                    .create_command_buffer()
                    .map_err(|err| AudioError::Runtime(format!("failed to create post_module command buffer: {err}")))?
                    .start_encoding();
                if !copied_output_prefix {
                    command_buffer.with_copy_encoder(|copy_encoder| {
                        let latent_buffer = latent_nsc.buffer();
                        let output_buffer = output.buffer();
                        let latent_buffer = latent_buffer.borrow();
                        let output_buffer = output_buffer.borrow();
                        copy_encoder.encode_copy_ranges(
                            (&latent_buffer, latent_nsc.offset()),
                            (&output_buffer, output.offset()),
                            full_copy_bytes,
                        );
                    });
                    copied_output_prefix = true;
                }
                let source = array_batch_view(latent_nsc, batch_index, frames, self.input_dim, active_len)?;
                command_buffer.with_copy_encoder(|copy_encoder| {
                    let source_buffer = source.buffer();
                    let main_output_buffer = main_output.buffer();
                    let source_buffer = source_buffer.borrow();
                    let main_output_buffer = main_output_buffer.borrow();
                    copy_encoder.encode_copy_ranges(
                        (&source_buffer, source.offset()),
                        (&main_output_buffer, main_output.offset()),
                        source.size(),
                    );
                });
                Self::encode_post_module_layers(&runtime, &mut state, &mut command_buffer)?;
                let destination = array_batch_view(&output, batch_index, frames, self.input_dim, active_len)?;
                command_buffer.with_copy_encoder(|copy_encoder| {
                    let main_output_buffer = main_output.buffer();
                    let destination_buffer = destination.buffer();
                    let main_output_buffer = main_output_buffer.borrow();
                    let destination_buffer = destination_buffer.borrow();
                    copy_encoder.encode_copy_ranges(
                        (&main_output_buffer, main_output.offset()),
                        (&destination_buffer, destination.offset()),
                        destination.size(),
                    );
                });
                let cpu_encode_ms = encode_start.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
                let command_buffer = command_buffer.end_encoding().submit();
                let wait_start = profile.is_some().then(Instant::now);
                let command_buffer = command_buffer.wait_until_completed().map_err(|err| {
                    AudioError::Runtime(format!("failed to wait for post_module command buffer: {err}"))
                })?;
                let cpu_wait_ms = wait_start.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
                let label = if batch_size == 1 && active_len == frames {
                    "post_module".to_string()
                } else {
                    format!("post_module_len_{active_len}_batch_{batch_index}")
                };
                push_audio_command_buffer_profile(profile, label, &command_buffer, cpu_encode_ms, cpu_wait_ms, None);
            }
        }

        Ok(output)
    }

}
