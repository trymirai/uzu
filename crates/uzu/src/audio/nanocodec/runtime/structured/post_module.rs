impl StructuredAudioCodecGraph {
    #[cfg(test)]
    fn apply_norm_sequence(
        values: &mut [f32],
        seq_len: usize,
        channels: usize,
        norm: &StructuredAudioNormLayer,
    ) -> AudioResult<()> {
        if norm.scales.len() != channels {
            return Err(AudioError::Runtime(format!(
                "norm scale length mismatch: expected {channels}, got {}",
                norm.scales.len()
            )));
        }
        if let Some(biases) = &norm.biases {
            if biases.len() != channels {
                return Err(AudioError::Runtime(format!(
                    "norm bias length mismatch: expected {channels}, got {}",
                    biases.len()
                )));
            }
        }
        let expected = checked_product(&[seq_len, channels])?;
        if values.len() != expected {
            return Err(AudioError::InvalidTokenShape {
                expected_tokens: expected,
                actual_tokens: values.len(),
            });
        }

        for token in 0..seq_len {
            let row_start = token * channels;
            let row = &mut values[row_start..row_start + channels];
            let mean = if norm.subtract_mean {
                row.iter().sum::<f32>() / channels as f32
            } else {
                0.0
            };
            let variance_sum = if norm.subtract_mean {
                row.iter()
                    .map(|&value| {
                        let centered = value - mean;
                        centered * centered
                    })
                    .sum::<f32>()
            } else {
                row.iter().map(|&value| value * value).sum::<f32>()
            };
            let variance = variance_sum / channels as f32;

            let inv_std = 1.0 / (variance + norm.epsilon).sqrt();
            for channel in 0..channels {
                let mut normalized = if norm.subtract_mean {
                    row[channel] - mean
                } else {
                    row[channel]
                };
                normalized *= inv_std * norm.scales[channel];
                if let Some(biases) = &norm.biases {
                    normalized += biases[channel];
                }
                row[channel] = normalized;
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn linear_sequence(
        input: &[f32],
        seq_len: usize,
        in_dim: usize,
        weight: &MatrixF32,
        bias: Option<&[f32]>,
    ) -> AudioResult<Vec<f32>> {
        let expected_input = checked_product(&[seq_len, in_dim])?;
        if input.len() != expected_input {
            return Err(AudioError::InvalidTokenShape {
                expected_tokens: expected_input,
                actual_tokens: input.len(),
            });
        }
        if weight.cols != in_dim {
            return Err(AudioError::Runtime(format!(
                "linear input dim mismatch: expected {}, got {}",
                weight.cols, in_dim
            )));
        }
        if let Some(bias_values) = bias {
            if bias_values.len() != weight.rows {
                return Err(AudioError::Runtime(format!(
                    "linear bias shape mismatch: expected {}, got {}",
                    weight.rows,
                    bias_values.len()
                )));
            }
        }

        let mut output = vec![0.0_f32; checked_product(&[seq_len, weight.rows])?];
        for token in 0..seq_len {
            let input_row = &input[token * in_dim..(token + 1) * in_dim];
            let output_row = &mut output[token * weight.rows..(token + 1) * weight.rows];
            for row_index in 0..weight.rows {
                let mut acc = bias.map_or(0.0, |bias_values| bias_values[row_index]);
                let row = &weight.values[row_index * weight.cols..(row_index + 1) * weight.cols];
                for (&weight_value, &input_value) in row.iter().zip(input_row.iter()) {
                    acc += weight_value * input_value;
                }
                output_row[row_index] = acc;
            }
        }
        Ok(output)
    }

    fn apply_convnext_ncs_enqueued(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
        command_buffer: &mut MetalCommandBuffer,
        input: &Array<Metal>,
        layer: &StructuredAudioConvNeXtGpuLayer,
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

    fn build_post_module_runtime(
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
            transformer_config: self.post_module_transformer_config.clone(),
            vocab_size: 1,
        };
        let decoder_config =
            Rc::new(inner_model_config.to_decoder_config().map_err(|_| {
                AudioError::Runtime("failed to build FishAudio post_module decoder config".to_string())
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
        let transformer_tree = root_loader_view
            .subtree(transformer_subtree_name)
            .map_err(|err| AudioError::Runtime(format!("missing FishAudio post_module subtree: {err}")))?;

        let max_sequence_length = decoder_config.context_length.max(required_sequence_length.max(1));
        let shared_buffers = Rc::new(RefCell::new(SharedBuffers::new(context.as_ref(), &decoder_config, &model_shape)));
        shared_buffers.borrow_mut().update_data_with_transformer_subtree(&root_loader_view, transformer_subtree_name);
        let scratch_buffers = ScratchBuffers::new(
            context.as_ref(),
            &decoder_config,
            &model_shape,
            max_sequence_length,
            max_sequence_length,
        );

        let attention_data_type = (0..decoder_config.num_layers).find_map(|layer_index| {
            let layer_config = decoder_config
                .layer_configs
                .as_ref()
                .map(|configs| &configs[layer_index])
                .unwrap_or(&decoder_config.layer_config);
            layer_config
                .attention_config()
                .map(|attention_config| attention_config.qkv_projection_config.activation_precision().into())
        });
        let attention_data_type =
            attention_data_type.ok_or(AudioError::Runtime("post_module has no attention layers".to_string()))?;

        let create_rope_block = |rope_type: RopeType| -> AudioResult<Rc<Rope<Metal>>> {
            let rope = Rope::<Metal>::new(context.as_ref(), attention_data_type, rope_type)
                .map_err(|err| AudioError::Runtime(format!("failed to initialize post_module rope block: {err}")))?;
            Ok(Rc::new(rope))
        };
        let global_rope =
            decoder_config.global_rope_config.as_ref().map(|_| create_rope_block(RopeType::Global)).transpose()?;

        let mut layers = Vec::with_capacity(decoder_config.num_layers);
        for layer_index in 0..decoder_config.num_layers {
            let layer_config = decoder_config
                .layer_configs
                .as_ref()
                .map(|configs| &configs[layer_index])
                .unwrap_or(&decoder_config.layer_config);
            let layer_type = model_shape.layer_type(layer_index);
            let rope_for_layer = match layer_type {
                crate::config::DecoderLayerType::Transformer => Some(
                    global_rope.clone().ok_or(AudioError::Runtime("post_module missing global rope".to_string()))?,
                ),
                _ => None,
            };
            let layer_loader = transformer_tree
                .subtree(&format!("layers.{layer_index}"))
                .map_err(|err| AudioError::Runtime(format!("failed to load post_module layer {layer_index}: {err}")))?;

            layers.push(LayerExecutables::new(
                context.clone(),
                layer_config,
                layer_type,
                layer_index,
                decoder_config.model_dim,
                decoder_config.hidden_dim,
                decoder_config.num_heads,
                decoder_config.head_dim,
                decoder_config.num_groups,
                decoder_config.attention_scale,
                &layer_loader,
                rope_for_layer,
            ));
        }

        let norm_reference_layer =
            decoder_config.layer_configs.as_ref().map(|configs| &configs[0]).unwrap_or(&decoder_config.layer_config);
        let norm_data_type: DataType = match &norm_reference_layer.mixer_config {
            crate::config::MixerConfig::Attention(attention_config) => {
                attention_config.qkv_projection_config.activation_precision().into()
            },
            crate::config::MixerConfig::Mamba(mamba_config) => {
                mamba_config.in_projection_config.activation_precision().into()
            },
            crate::config::MixerConfig::ShortConv(short_conv_config) => {
                short_conv_config.in_projection_config.activation_precision().into()
            },
        };
        let output_norm_tree = transformer_tree
            .subtree("output_norm")
            .map_err(|err| AudioError::Runtime(format!("failed to load post_module output_norm: {err}")))?;
        let output_norm = RMSNorm::new(
            context.as_ref(),
            norm_data_type,
            decoder_config.output_norm_config.clone(),
            ArrayId::Main,
            ArrayId::Main,
            &output_norm_tree,
        )
        .map_err(|err| AudioError::Runtime(format!("failed to build post_module output_norm: {err}")))?;

        Ok(StructuredAudioPostModuleRuntime {
            context,
            model_shape,
            scratch_buffers,
            shared_buffers,
            layers: layers.into_boxed_slice(),
            output_norm,
            max_sequence_length,
        })
    }

    fn post_module_runtime_on_context(
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

    fn decode_context(&self) -> AudioResult<Rc<<Metal as Backend>::Context>> {
        FISHAUDIO_DECODE_CONTEXT_CACHE.with(|cache| {
            if let Some(existing) = cache.borrow().get(&self.weights_path).cloned() {
                return Ok(existing);
            }
            let created = <Metal as Backend>::Context::new()
                .map_err(|err| AudioError::Runtime(format!("failed to create FishAudio decode context: {err}")))?;
            cache.borrow_mut().insert(self.weights_path.clone(), created.clone());
            Ok(created)
        })
    }

    fn build_quantizer_gpu_resources(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
    ) -> AudioResult<FishAudioQuantizerGpuResources> {
        if self.semantic_codebook_size == 0 || self.codebook_size == 0 {
            return Err(AudioError::InvalidTokenCardinality);
        }
        let codebook_dim = self.semantic_quantizer.codebook.cols;
        if codebook_dim == 0 {
            return Err(AudioError::InvalidTokenCardinality);
        }
        if self.semantic_quantizer.codebook.rows != self.semantic_codebook_size {
            return Err(AudioError::Runtime(format!(
                "semantic codebook row mismatch: expected {}, got {}",
                self.semantic_codebook_size, self.semantic_quantizer.codebook.rows
            )));
        }
        if self.semantic_quantizer.out_proj.rows != self.input_dim
            || self.semantic_quantizer.out_proj.cols != codebook_dim
        {
            return Err(AudioError::Runtime("semantic out_proj shape mismatch".to_string()));
        }
        if self.semantic_quantizer.out_bias.len() != self.input_dim {
            return Err(AudioError::Runtime("semantic out_bias shape mismatch".to_string()));
        }

        let residual_quantizers = self.residual_quantizers.len();
        for (index, quantizer) in self.residual_quantizers.iter().enumerate() {
            if quantizer.codebook.rows != self.codebook_size || quantizer.codebook.cols != codebook_dim {
                return Err(AudioError::Runtime(format!("residual quantizer {index} codebook shape mismatch")));
            }
            if quantizer.out_proj.rows != self.input_dim || quantizer.out_proj.cols != codebook_dim {
                return Err(AudioError::Runtime(format!("residual quantizer {index} out_proj shape mismatch")));
            }
            if quantizer.out_bias.len() != self.input_dim {
                return Err(AudioError::Runtime(format!("residual quantizer {index} out_bias shape mismatch")));
            }
        }

        let data_type = self.vocoder_data_type;
        let kernel =
            <<Metal as Backend>::Kernels as Kernels>::AudioQuantizerDecodeKernel::new(context.as_ref(), data_type)
                .map_err(|err| AudioError::Runtime(format!("failed to initialize quantizer decode kernel: {err}")))?;

        let mut semantic_codebook = context.create_array(
            &[self.semantic_codebook_size, codebook_dim],
            data_type,
            "fishaudio_quantizer_semantic_codebook",
        );
        write_f32_slice_to_array(&mut semantic_codebook, &self.semantic_quantizer.codebook.values)?;

        let mut semantic_out_proj =
            context.create_array(&[self.input_dim, codebook_dim], data_type, "fishaudio_quantizer_semantic_out_proj");
        write_f32_slice_to_array(&mut semantic_out_proj, &self.semantic_quantizer.out_proj.values)?;

        let mut semantic_out_bias =
            context.create_array(&[self.input_dim], data_type, "fishaudio_quantizer_semantic_out_bias");
        write_f32_slice_to_array(&mut semantic_out_bias, &self.semantic_quantizer.out_bias)?;

        let residual_count_for_shape = residual_quantizers.max(1);
        let residual_codebook_rows_for_shape = self.codebook_size.max(1);
        let residual_codebook_len =
            checked_product(&[residual_count_for_shape, residual_codebook_rows_for_shape, codebook_dim])?;
        let residual_proj_len = checked_product(&[residual_count_for_shape, self.input_dim, codebook_dim])?;
        let residual_bias_len = checked_product(&[residual_count_for_shape, self.input_dim])?;
        let mut residual_codebook_host = vec![0.0_f32; residual_codebook_len];
        let mut residual_proj_host = vec![0.0_f32; residual_proj_len];
        let mut residual_bias_host = vec![0.0_f32; residual_bias_len];
        for (index, quantizer) in self.residual_quantizers.iter().enumerate() {
            let codebook_offset = index
                .checked_mul(self.codebook_size)
                .and_then(|value| value.checked_mul(codebook_dim))
                .ok_or(AudioError::Runtime("residual codebook offset overflow".to_string()))?;
            let codebook_end = codebook_offset
                .checked_add(checked_product(&[self.codebook_size, codebook_dim])?)
                .ok_or(AudioError::Runtime("residual codebook offset overflow".to_string()))?;
            residual_codebook_host[codebook_offset..codebook_end].copy_from_slice(&quantizer.codebook.values);

            let proj_offset = index
                .checked_mul(self.input_dim)
                .and_then(|value| value.checked_mul(codebook_dim))
                .ok_or(AudioError::Runtime("residual proj offset overflow".to_string()))?;
            let proj_end = proj_offset
                .checked_add(checked_product(&[self.input_dim, codebook_dim])?)
                .ok_or(AudioError::Runtime("residual proj offset overflow".to_string()))?;
            residual_proj_host[proj_offset..proj_end].copy_from_slice(&quantizer.out_proj.values);

            let bias_offset = index
                .checked_mul(self.input_dim)
                .ok_or(AudioError::Runtime("residual bias offset overflow".to_string()))?;
            let bias_end = bias_offset
                .checked_add(self.input_dim)
                .ok_or(AudioError::Runtime("residual bias offset overflow".to_string()))?;
            residual_bias_host[bias_offset..bias_end].copy_from_slice(&quantizer.out_bias);
        }

        let mut residual_codebooks = context.create_array(
            &[residual_count_for_shape, residual_codebook_rows_for_shape, codebook_dim],
            data_type,
            "fishaudio_quantizer_residual_codebooks",
        );
        write_f32_slice_to_array(&mut residual_codebooks, &residual_codebook_host)?;

        let mut residual_out_proj = context.create_array(
            &[residual_count_for_shape, self.input_dim, codebook_dim],
            data_type,
            "fishaudio_quantizer_residual_out_proj",
        );
        write_f32_slice_to_array(&mut residual_out_proj, &residual_proj_host)?;

        let mut residual_out_bias = context.create_array(
            &[residual_count_for_shape, self.input_dim],
            data_type,
            "fishaudio_quantizer_residual_out_bias",
        );
        write_f32_slice_to_array(&mut residual_out_bias, &residual_bias_host)?;

        Ok(FishAudioQuantizerGpuResources {
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

    fn quantizer_gpu_resources(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
    ) -> AudioResult<Rc<FishAudioQuantizerGpuResources>> {
        let key = ((Rc::as_ptr(context) as usize) << 8) | usize::from(fishaudio_dtype_key(self.vocoder_data_type));
        FISHAUDIO_QUANTIZER_RESOURCES_CACHE.with(|cache| {
            if let Some(existing) = cache.borrow().get(&key) {
                return Ok(existing.clone());
            }
            let resources = Rc::new(self.build_quantizer_gpu_resources(context)?);
            cache.borrow_mut().insert(key, resources.clone());
            Ok(resources)
        })
    }

    fn build_vocoder_gpu_graph(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
    ) -> AudioResult<StructuredAudioDecoderGpuGraph> {
        let data_type = self.vocoder_data_type;
        let first_conv =
            create_conv1d_gpu_layer(context, data_type, &self.decoder.first_conv, "fishaudio_decoder_first_conv")?;
        let final_snake_alpha = create_alpha_gpu_array(
            context,
            data_type,
            self.decoder.final_conv.cin,
            &self.decoder.final_snake_alpha,
            "fishaudio_decoder_final_snake_alpha",
        )?;
        let final_conv =
            create_conv1d_gpu_layer(context, data_type, &self.decoder.final_conv, "fishaudio_decoder_final_conv")?;

        let mut upsample_blocks = Vec::with_capacity(self.decoder.upsample_blocks.len());
        for (index, (trans_conv, convnext)) in self.decoder.upsample_blocks.iter().enumerate() {
            let trans_conv = create_conv_transpose1d_gpu_layer(
                context,
                data_type,
                trans_conv,
                &format!("fishaudio_upsample_{index}_trans_conv"),
            )?;
            let convnext = create_convnext_gpu_layer(
                context,
                data_type,
                convnext,
                &format!("fishaudio_upsample_{index}_convnext"),
            )?;
            upsample_blocks.push((trans_conv, convnext));
        }

        let mut decoder_blocks = Vec::with_capacity(self.decoder.decoder_blocks.len());
        for (index, block) in self.decoder.decoder_blocks.iter().enumerate() {
            let channels = block.trans_conv.cout;
            let snake_alpha = create_alpha_gpu_array(
                context,
                data_type,
                block.trans_conv.cin,
                &block.snake_alpha,
                &format!("fishaudio_decoder_block_{index}_snake_alpha"),
            )?;
            let trans_conv = create_conv_transpose1d_gpu_layer(
                context,
                data_type,
                &block.trans_conv,
                &format!("fishaudio_decoder_block_{index}_trans_conv"),
            )?;
            let res_unit1 = create_residual_unit_gpu_layer(
                context,
                data_type,
                &block.res_unit1,
                channels,
                &format!("fishaudio_decoder_block_{index}_res1"),
            )?;
            let res_unit2 = create_residual_unit_gpu_layer(
                context,
                data_type,
                &block.res_unit2,
                channels,
                &format!("fishaudio_decoder_block_{index}_res2"),
            )?;
            let res_unit3 = create_residual_unit_gpu_layer(
                context,
                data_type,
                &block.res_unit3,
                channels,
                &format!("fishaudio_decoder_block_{index}_res3"),
            )?;
            decoder_blocks.push(StructuredAudioDecoderBlockGpuLayer {
                snake_alpha,
                trans_conv,
                res_unit1,
                res_unit2,
                res_unit3,
            });
        }

        Ok(StructuredAudioDecoderGpuGraph {
            first_conv,
            upsample_blocks,
            decoder_blocks,
            final_snake_alpha,
            final_conv,
        })
    }

    fn vocoder_gpu_graph(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
    ) -> AudioResult<Rc<StructuredAudioDecoderGpuGraph>> {
        let key = ((Rc::as_ptr(context) as usize) << 8) | usize::from(fishaudio_dtype_key(self.vocoder_data_type));
        FISHAUDIO_VOCODER_GRAPH_CACHE.with(|cache| {
            if let Some(existing) = cache.borrow().get(&key) {
                return Ok(existing.clone());
            }
            let graph = Rc::new(self.build_vocoder_gpu_graph(context)?);
            cache.borrow_mut().insert(key, graph.clone());
            Ok(graph)
        })
    }

    fn encode_post_module_layers(
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

    fn apply_post_module_gpu_on_array_single_batch(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
        latent_nsc: &Array<Metal>,
        frames: usize,
        profile: &mut Option<AudioDecodeProfile>,
    ) -> AudioResult<Array<Metal>> {
        if self.post_module_model_dim != self.input_dim {
            return Err(AudioError::Runtime("post_module model_dim mismatch".to_string()));
        }
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

    fn apply_post_module_gpu_on_array(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
        latent_nsc: &Array<Metal>,
        lengths: &[usize],
        batch_size: usize,
        frames: usize,
        profile: &mut Option<AudioDecodeProfile>,
    ) -> AudioResult<Array<Metal>> {
        if self.post_module_model_dim != self.input_dim {
            return Err(AudioError::Runtime("post_module model_dim mismatch".to_string()));
        }
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
            "fishaudio_post_module_output_nsc",
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
