use super::*;

fn copy_array_into_allocation<B: Backend>(
    encoder: &mut Encoder<B>,
    source: &Array<B>,
    destination: &mut crate::backends::common::Allocation<B>,
) {
    let source_buffer = source.buffer();
    let source_buffer = source_buffer.borrow();
    let (destination_buffer, destination_range) = destination.as_buffer_range();
    encoder.encode_copy(
        &source_buffer,
        source.offset()..source.offset() + source.size(),
        destination_buffer,
        destination_range.start..destination_range.start + source.size(),
    );
}

fn copy_allocation_into_array<B: Backend>(
    encoder: &mut Encoder<B>,
    source: &crate::backends::common::Allocation<B>,
    destination: &Array<B>,
) {
    let (source_buffer, source_range) = source.as_buffer_range();
    let destination_buffer = destination.buffer();
    let mut destination_buffer = destination_buffer.borrow_mut();
    encoder.encode_copy(
        source_buffer,
        source_range.start..source_range.start + destination.size(),
        &mut destination_buffer,
        destination.offset()..destination.offset() + destination.size(),
    );
}

impl StructuredAudioCodecGraph {
    pub(super) fn apply_convnext_ncs_enqueued<B: Backend>(
        &self,
        resources: &StructuredAudioRuntimeResources<B>,
        encoder: &mut Encoder<B>,
        input: &Array<B>,
        layer: &StructuredAudioConvNeXt<B>,
        lengths: &[i32],
        lengths_array: &Array<B>,
        batch_size: usize,
        channels: usize,
        seq_len: usize,
        kernels: &StructuredAudioKernelCache<B>,
    ) -> AudioResult<Array<B>> {
        let ws = &resources.decode_workspace;
        let ctx = resources.context();
        let data_type = input.data_type();

        let residual = input.clone();
        let x = causal_conv1d_grouped_enqueue(
            encoder,
            input,
            ws.next_scratch(ctx, &[batch_size, channels, seq_len], data_type, "cnxt_dw"),
            &layer.depthwise_conv,
            SequenceLayout::Ncs,
            lengths,
            lengths_array,
            batch_size,
            seq_len,
            kernels,
        )?;
        let x = norm_ncs_enqueue(
            encoder,
            &x,
            ws.next_scratch(ctx, &[batch_size, channels, seq_len], data_type, "cnxt_norm"),
            &layer.norm,
            lengths,
            lengths_array,
            batch_size,
            channels,
            seq_len,
            kernels,
        )?;
        let x = conv1d_pointwise_ncs_enqueue(
            encoder,
            &x,
            ws.next_scratch(ctx, &[batch_size, layer.pwconv1.cout, seq_len], data_type, "cnxt_pw1"),
            &layer.pwconv1,
            lengths,
            lengths_array,
            batch_size,
            seq_len,
            kernels,
        )?;
        let x = gelu_enqueue(encoder, &x, ws.next_scratch(ctx, x.shape(), data_type, "cnxt_gelu"), kernels)?;
        let x = conv1d_pointwise_ncs_enqueue(
            encoder,
            &x,
            ws.next_scratch(ctx, &[batch_size, layer.pwconv2.cout, seq_len], data_type, "cnxt_pw2"),
            &layer.pwconv2,
            lengths,
            lengths_array,
            batch_size,
            seq_len,
            kernels,
        )?;
        add_enqueue(encoder, &x, &residual, ws.next_scratch(ctx, x.shape(), data_type, "cnxt_add"), kernels)
    }

    pub(super) fn build_post_module_runtime<B: Backend>(
        &self,
        context: Rc<B::Context>,
        required_sequence_length: usize,
    ) -> AudioResult<StructuredAudioPostModuleRuntime<B>> {
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
        let decoder_config = Rc::new(inner_model_config.to_decoder_config().map_err(|_| {
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
        let mut shared_buffers = SharedBuffers::new(context.as_ref(), &decoder_config, &model_shape);
        {
            let transformer_tree = root_loader_view
                .subtree(transformer_subtree_name)
                .map_err(|err| AudioError::Runtime(format!("missing structured audio post_module subtree: {err}")))?;
            if let Some(global_rope) = &mut shared_buffers.global_rope {
                global_rope.update_data(&transformer_tree, "global_rope");
            }
            if let Some(local_rope) = &mut shared_buffers.local_rope {
                local_rope.update_data(&transformer_tree, "local_rope");
            }
        }
        let shared_buffers = Rc::new(shared_buffers);
        let (layers, output_norm) = Decoder::build_transformer_layers_and_norm(
            context.as_ref(),
            &decoder_config,
            &root_loader_view,
            transformer_subtree_name,
        );

        Ok(StructuredAudioPostModuleRuntime {
            context,
            model_shape,
            shared_buffers,
            layers,
            output_norm,
            max_sequence_length,
        })
    }

    pub(super) fn post_module_runtime<B: Backend>(
        &self,
        resources: &StructuredAudioRuntimeResources<B>,
        required_sequence_length: usize,
    ) -> AudioResult<Rc<StructuredAudioPostModuleRuntime<B>>> {
        {
            let cached = resources.post_module_runtime.borrow();
            if let Some(runtime) = cached.as_ref() {
                if runtime.max_sequence_length >= required_sequence_length.max(1) {
                    return Ok(runtime.clone());
                }
            }
        }
        let runtime = Rc::new(self.build_post_module_runtime(resources.context().clone(), required_sequence_length)?);
        *resources.post_module_runtime.borrow_mut() = Some(runtime.clone());
        Ok(runtime)
    }

    pub(super) fn build_quantizer_resources<B: Backend>(
        &self,
        context: &Rc<B::Context>,
    ) -> AudioResult<FishAudioQuantizerResources<B>> {
        if self.semantic_codebook_size == 0 || self.codebook_size == 0 {
            return Err(AudioError::InvalidTokenCardinality);
        }
        let data_type = self.vocoder_data_type;
        let kernel = <B::Kernels as Kernels>::AudioQuantizerDecodeKernel::new(context.as_ref(), data_type)
            .map_err(|err| AudioError::Runtime(format!("failed to initialize quantizer decode kernel: {err}")))?;
        let weights_file = File::open(self.weights_path.as_str()).map_err(|err| {
            AudioError::Runtime(format!("failed to open structured audio weights '{}': {err}", self.weights_path))
        })?;
        let loader = ParameterLoader::new(&weights_file, context.as_ref()).map_err(|err| {
            AudioError::Runtime(format!("failed to load structured audio weights '{}': {err}", self.weights_path))
        })?;
        let root = loader.tree();
        let audio_decoder_tree = root.subtree("audio_decoder")?;
        let quantizer_tree = audio_decoder_tree.subtree("quantizer")?;
        let semantic_tree = quantizer_tree.subtree("semantic_quantizer")?.subtree("quantizers")?.subtree("0")?;
        let semantic_codebook = read_float_matrix_exact::<B>(
            &semantic_tree.subtree("codebook")?,
            "weights",
            self.semantic_codebook_size,
            self.config.codebook_dim,
            data_type,
        )?;
        let codebook_dim = semantic_codebook.shape()[1];
        let semantic_out_proj = read_float_matrix_exact::<B>(
            &semantic_tree.subtree("out_proj")?,
            "weights",
            self.input_dim,
            codebook_dim,
            data_type,
        )?;
        let semantic_out_bias =
            read_float_vector_exact::<B>(&semantic_tree.subtree("out_proj")?, "biases", self.input_dim, data_type)?;

        let residual_quantizers = self.config.n_codebooks;
        let residual_count_for_shape = residual_quantizers.max(1);
        let residual_codebook_rows_for_shape = self.codebook_size.max(1);
        let residual_codebooks = context.create_array_zeros(
            &[residual_count_for_shape, residual_codebook_rows_for_shape, codebook_dim],
            data_type,
            "structured_audio_quantizer_residual_codebooks",
        );
        let residual_out_proj = context.create_array_zeros(
            &[residual_count_for_shape, self.input_dim, codebook_dim],
            data_type,
            "structured_audio_quantizer_residual_out_proj",
        );
        let residual_out_bias = context.create_array_zeros(
            &[residual_count_for_shape, self.input_dim],
            data_type,
            "structured_audio_quantizer_residual_out_bias",
        );

        let residual_root = quantizer_tree.subtree("quantizer")?.subtree("quantizers")?;
        for index in 0..residual_quantizers {
            let quantizer_tree = residual_root.subtree(&index.to_string())?;
            let codebook = read_float_matrix_exact::<B>(
                &quantizer_tree.subtree("codebook")?,
                "weights",
                self.codebook_size,
                codebook_dim,
                data_type,
            )?;
            let out_proj = read_float_matrix_exact::<B>(
                &quantizer_tree.subtree("out_proj")?,
                "weights",
                self.input_dim,
                codebook_dim,
                data_type,
            )?;
            let out_bias = read_float_vector_exact::<B>(
                &quantizer_tree.subtree("out_proj")?,
                "biases",
                self.input_dim,
                data_type,
            )?;

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

    pub(super) fn quantizer_resources<B: Backend>(
        &self,
        resources: &StructuredAudioRuntimeResources<B>,
    ) -> AudioResult<Rc<FishAudioQuantizerResources<B>>> {
        if let Some(existing) = resources.quantizer_resources.borrow().as_ref() {
            return Ok(existing.clone());
        }
        let created = Rc::new(self.build_quantizer_resources(resources.context())?);
        *resources.quantizer_resources.borrow_mut() = Some(created.clone());
        Ok(created)
    }

    pub(super) fn build_vocoder_graph<B: Backend>(
        &self,
        context: &Rc<B::Context>,
    ) -> AudioResult<StructuredAudioDecoderGraph<B>> {
        let weights_file = File::open(self.weights_path.as_str()).map_err(|err| {
            AudioError::Runtime(format!("failed to open structured audio weights '{}': {err}", self.weights_path))
        })?;
        let loader = ParameterLoader::new(&weights_file, context.as_ref()).map_err(|err| {
            AudioError::Runtime(format!("failed to load structured audio weights '{}': {err}", self.weights_path))
        })?;
        let root = loader.tree();
        build_vocoder_graph_from_tree::<B>(context, &root, &self.config, self.vocoder_data_type)
    }

    pub(super) fn vocoder_graph<B: Backend>(
        &self,
        resources: &StructuredAudioRuntimeResources<B>,
    ) -> AudioResult<Rc<StructuredAudioDecoderGraph<B>>> {
        if let Some(existing) = resources.vocoder_graph.borrow().as_ref() {
            return Ok(existing.clone());
        }
        let created = Rc::new(self.build_vocoder_graph(resources.context())?);
        *resources.vocoder_graph.borrow_mut() = Some(created.clone());
        Ok(created)
    }

    pub(super) fn encode_post_module_layers<B: Backend>(
        runtime: &StructuredAudioPostModuleRuntime<B>,
        state: &mut ForwardPassState<B>,
        main: crate::backends::common::Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> AudioResult<crate::backends::common::Allocation<B>> {
        let encoding_parameters = EncodingParameters::new();
        let mut main = main;
        let mut shortcut = encoder
            .allocate_scratch(main.as_buffer_range().1.len())
            .map_err(|err| AudioError::Runtime(format!("post_module shortcut allocation failed: {err}")))?;
        for layer in runtime.layers.iter() {
            let rope_type = layer.rope_type();
            let rope_cosines = rope_type.map(|rope_type| state.rope_cosines(rope_type));
            let rope_sines = rope_type.map(|rope_type| state.rope_sines(rope_type));
            #[cfg(feature = "tracing")]
            let trace = state.traces().borrow().layer_results.get(layer.layer_index).cloned();
            main = if state.cache_layers().is_some() {
                state
                    .with_cache_layer_mut(layer.layer_index, |cache_layer| {
                        layer.encode(
                            crate::encodable_block::LayerArguments {
                                context: state.context(),
                                batch_dim: state.active_row_count(),
                                token_positions: state.token_positions(),
                                token_parents: state.token_parents(),
                                token_subtrie_ranges: state.token_subtrie_ranges(),
                                attention_sinks: state.attention_sinks(layer.layer_index),
                                rope_cosines,
                                rope_sines,
                                rope_max_sequence_length: state.rope_max_sequence_length(),
                                rope_dim: state.rope_dim(),
                                sampling_start: state.sampling_start(),
                                sampling_length: state.sampling_length(),
                                cache_layer: Some(cache_layer),
                                #[cfg(feature = "tracing")]
                                trace,
                            },
                            &encoding_parameters,
                            main,
                            &mut shortcut,
                            encoder,
                        )
                    })
                    .map_err(|err| AudioError::Runtime(format!("post_module layer encode failed: {err}")))?
            } else {
                layer
                    .encode(
                        crate::encodable_block::LayerArguments {
                            context: state.context(),
                            batch_dim: state.active_row_count(),
                            token_positions: state.token_positions(),
                            token_parents: state.token_parents(),
                            token_subtrie_ranges: state.token_subtrie_ranges(),
                            attention_sinks: state.attention_sinks(layer.layer_index),
                            rope_cosines,
                            rope_sines,
                            rope_max_sequence_length: state.rope_max_sequence_length(),
                            rope_dim: state.rope_dim(),
                            sampling_start: state.sampling_start(),
                            sampling_length: state.sampling_length(),
                            cache_layer: None,
                            #[cfg(feature = "tracing")]
                            trace,
                        },
                        &encoding_parameters,
                        main,
                        &mut shortcut,
                        encoder,
                    )
                    .map_err(|err| AudioError::Runtime(format!("post_module layer encode failed: {err}")))?
            };
        }
        runtime
            .output_norm
            .encode(&main, 0, state.active_row_count(), Some(&mut shortcut), encoder)
            .map_err(|err| AudioError::Runtime(format!("post_module output norm encode failed: {err}")))
    }

    pub(super) fn apply_post_module_single_batch_enqueued<B: Backend>(
        &self,
        resources: &StructuredAudioRuntimeResources<B>,
        encoder: &mut Encoder<B>,
        latent_nsc: &Array<B>,
        frames: usize,
    ) -> AudioResult<Array<B>> {
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

        let runtime = self.post_module_runtime(resources, frames.max(1))?;
        let token_ids = vec![0_u64; frames];
        let token_positions = (0..frames).collect::<Vec<_>>();
        let mut state = ForwardPassState::new_classifier(
            runtime.context.clone(),
            &runtime.model_shape,
            runtime.shared_buffers.clone(),
            &token_ids,
            &token_positions,
            1,
        );

        let main_shape = runtime.model_shape.main_shape(frames);
        if main_shape != [frames, self.input_dim] {
            return Err(AudioError::Runtime(format!(
                "post_module main shape mismatch: expected [{frames}, {}], got [{}, {}]",
                self.input_dim, main_shape[0], main_shape[1]
            )));
        }
        if runtime.model_shape.activation_data_type() != latent_nsc.data_type() {
            return Err(AudioError::Runtime(format!(
                "post_module dtype mismatch: main={:?}, latent={:?}",
                runtime.model_shape.activation_data_type(),
                latent_nsc.data_type()
            )));
        }

        let main_output = runtime.context.create_array_zeros(
            &[frames, self.input_dim],
            latent_nsc.data_type(),
            "structured_audio_post_module_main",
        );
        let mut main = crate::forward_pass::state::allocation_helpers::create_allocation(
            runtime.context.as_ref(),
            &[frames, self.input_dim],
            latent_nsc.data_type(),
        );
        copy_array_into_allocation(encoder, latent_nsc, &mut main);
        let main = Self::encode_post_module_layers(&runtime, &mut state, main, encoder)?;
        copy_allocation_into_array(encoder, &main, &main_output);

        Ok(main_output)
    }

    pub(super) fn apply_post_module_enqueued<B: Backend>(
        &self,
        resources: &StructuredAudioRuntimeResources<B>,
        context: &Rc<B::Context>,
        encoder: &mut Encoder<B>,
        latent_nsc: &Array<B>,
        lengths: &[usize],
        batch_size: usize,
        frames: usize,
    ) -> AudioResult<Array<B>> {
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
            return self.apply_post_module_single_batch_enqueued(resources, encoder, latent_nsc, frames);
        }

        let output = context.create_array_zeros(
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
            let runtime = self.post_module_runtime(resources, active_len.max(1))?;
            let token_ids = vec![0_u64; active_len];
            let token_positions = (0..active_len).collect::<Vec<_>>();
            let mut state = ForwardPassState::new_classifier(
                runtime.context.clone(),
                &runtime.model_shape,
                runtime.shared_buffers.clone(),
                &token_ids,
                &token_positions,
                1,
            );

            let main_shape = runtime.model_shape.main_shape(active_len);
            if main_shape != [active_len, self.input_dim] {
                return Err(AudioError::Runtime(format!(
                    "post_module main shape mismatch: expected [{active_len}, {}], got [{}, {}]",
                    self.input_dim, main_shape[0], main_shape[1]
                )));
            }
            if runtime.model_shape.activation_data_type() != latent_nsc.data_type() {
                return Err(AudioError::Runtime(format!(
                    "post_module dtype mismatch: main={:?}, latent={:?}",
                    runtime.model_shape.activation_data_type(),
                    latent_nsc.data_type()
                )));
            }

            for &batch_index in &batch_indices {
                if !copied_output_prefix {
                    {
                        let latent_buffer = latent_nsc.buffer();
                        let output_buffer = output.buffer();
                        let latent_buffer = latent_buffer.borrow();
                        let mut output_buffer = output_buffer.borrow_mut();
                        encoder.encode_copy(
                            &latent_buffer,
                            latent_nsc.offset()..latent_nsc.offset() + full_copy_bytes,
                            &mut output_buffer,
                            output.offset()..output.offset() + full_copy_bytes,
                        );
                    }
                    copied_output_prefix = true;
                }
                let source = array_batch_view(latent_nsc, batch_index, frames, self.input_dim, active_len)?;
                let mut main = crate::forward_pass::state::allocation_helpers::create_allocation(
                    runtime.context.as_ref(),
                    &[active_len, self.input_dim],
                    source.data_type(),
                );
                copy_array_into_allocation(encoder, &source, &mut main);
                let main = Self::encode_post_module_layers(&runtime, &mut state, main, encoder)?;
                let destination = array_batch_view(&output, batch_index, frames, self.input_dim, active_len)?;
                copy_allocation_into_array(encoder, &main, &destination);
            }
        }

        Ok(output)
    }
}
