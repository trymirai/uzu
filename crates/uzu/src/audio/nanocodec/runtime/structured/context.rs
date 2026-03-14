impl StructuredAudioCodecGraph {
    fn conv1d_input_context(layer: &StructuredAudioConv1dLayer) -> AudioResult<usize> {
        layer
            .kernel_size
            .checked_sub(1)
            .and_then(|value| value.checked_mul(layer.dilation))
            .ok_or(AudioError::Runtime("conv1d streaming context overflow".to_string()))
    }

    fn convtranspose_input_context(
        output_context: usize,
        layer: &StructuredAudioConvTranspose1dLayer,
    ) -> AudioResult<usize> {
        let kernel_minus_one = layer.kernel_size.saturating_sub(1);
        let numerator = output_context
            .checked_add(kernel_minus_one)
            .ok_or(AudioError::Runtime("transpose streaming context overflow".to_string()))?;
        checked_div_ceil(numerator, layer.stride)
    }

    fn residual_unit_input_context(
        output_context: usize,
        layer: &StructuredAudioResidualUnitLayer,
    ) -> AudioResult<usize> {
        let conv2 = Self::conv1d_input_context(&layer.conv2)?;
        let conv1 = Self::conv1d_input_context(&layer.conv1)?;
        output_context
            .checked_add(conv2)
            .and_then(|value| value.checked_add(conv1))
            .ok_or(AudioError::Runtime("residual-unit streaming context overflow".to_string()))
    }

    fn convnext_input_context(
        output_context: usize,
        layer: &StructuredAudioConvNeXtLayer,
    ) -> AudioResult<usize> {
        output_context
            .checked_add(Self::conv1d_input_context(&layer.depthwise_conv)?)
            .ok_or(AudioError::Runtime("convnext streaming context overflow".to_string()))
    }

    fn decoder_block_input_context(
        output_context: usize,
        layer: &StructuredAudioDecoderBlockLayer,
    ) -> AudioResult<usize> {
        let after_res3 = Self::residual_unit_input_context(output_context, &layer.res_unit3)?;
        let after_res2 = Self::residual_unit_input_context(after_res3, &layer.res_unit2)?;
        let after_res1 = Self::residual_unit_input_context(after_res2, &layer.res_unit1)?;
        Self::convtranspose_input_context(after_res1, &layer.trans_conv)
    }

    fn streaming_vocoder_context_frames(&self) -> AudioResult<usize> {
        let mut context = Self::conv1d_input_context(&self.decoder.final_conv)?;

        for block in self.decoder.decoder_blocks.iter().rev() {
            context = Self::decoder_block_input_context(context, block)?;
        }

        context = context
            .checked_add(Self::conv1d_input_context(&self.decoder.first_conv)?)
            .ok_or(AudioError::Runtime("decoder-first streaming context overflow".to_string()))?;

        for (trans_conv, convnext) in self.decoder.upsample_blocks.iter().rev() {
            context = Self::convnext_input_context(context, convnext)?;
            context = Self::convtranspose_input_context(context, trans_conv)?;
        }

        Ok(context)
    }

    fn post_module_streaming_context_frames(&self) -> Option<usize> {
        let mut context = 0usize;
        for layer in &self.post_module_transformer_config.layer_configs {
            let window = layer.mixer_config.sliding_window_size()?;
            context = context.max(window.saturating_sub(1));
        }
        Some(context)
    }

    fn streaming_decode_context_frames(&self) -> AudioResult<Option<usize>> {
        let vocoder_context = self.streaming_vocoder_context_frames()?;
        let Some(post_module_context) = self.post_module_streaming_context_frames() else {
            return Ok(None);
        };
        Ok(Some(vocoder_context.max(post_module_context)))
    }

    #[cfg(test)]
    fn add_quantized_code_to_latent(
        target: &mut [f32],
        quantizer: &StructuredAudioVectorQuantizer,
        code_index: usize,
    ) -> AudioResult<()> {
        let code_row = quantizer
            .codebook
            .row(code_index)
            .ok_or(AudioError::Runtime("quantizer code index out of range".to_string()))?;
        if quantizer.out_proj.cols != code_row.len() || quantizer.out_proj.rows != target.len() {
            return Err(AudioError::Runtime("quantizer projection shape mismatch".to_string()));
        }
        for (row_index, row) in quantizer.out_proj.values.chunks_exact(quantizer.out_proj.cols).enumerate() {
            let mut acc = quantizer.out_bias[row_index];
            for (&w, &x) in row.iter().zip(code_row.iter()) {
                acc += w * x;
            }
            target[row_index] += acc;
        }
        Ok(())
    }

    #[cfg(test)]
    fn decode_quantizer_to_nsc_reference(
        &self,
        tokens: &[u32],
        lengths: &[usize],
        batch_size: usize,
        codebooks: usize,
        frames: usize,
    ) -> AudioResult<Vec<f32>> {
        if codebooks != self.total_codebooks {
            return Err(AudioError::Runtime(format!(
                "FishAudio codebook mismatch: expected {}, got {codebooks}",
                self.total_codebooks
            )));
        }
        let expected_tokens = checked_product(&[batch_size, codebooks, frames])?;
        if tokens.len() != expected_tokens {
            return Err(AudioError::InvalidTokenShape {
                expected_tokens,
                actual_tokens: tokens.len(),
            });
        }
        if lengths.len() != batch_size {
            return Err(AudioError::InvalidTokenLengths {
                expected_lengths: batch_size,
                actual_lengths: lengths.len(),
            });
        }

        let mut latent_nsc = vec![0.0_f32; checked_product(&[batch_size, frames, self.input_dim])?];
        for batch in 0..batch_size {
            let active_frames = lengths[batch];
            if active_frames > frames {
                return Err(AudioError::InvalidTokenLengthValue {
                    length: active_frames,
                    frames,
                });
            }
            for frame in 0..active_frames {
                let row_start = (batch * frames + frame) * self.input_dim;
                let row_end = row_start + self.input_dim;
                let target = &mut latent_nsc[row_start..row_end];

                let semantic_token_index = ((batch * codebooks) * frames) + frame;
                let semantic_token = tokens[semantic_token_index] as usize;
                let semantic_clamped = semantic_token.min(self.semantic_codebook_size.saturating_sub(1));
                Self::add_quantized_code_to_latent(target, &self.semantic_quantizer, semantic_clamped)?;

                for (residual_index, quantizer) in self.residual_quantizers.iter().enumerate() {
                    let token_index = ((batch * codebooks + (residual_index + 1)) * frames) + frame;
                    let token = tokens[token_index] as usize;
                    let clamped = token.min(self.codebook_size.saturating_sub(1));
                    Self::add_quantized_code_to_latent(target, quantizer, clamped)?;
                }
            }
        }

        Ok(latent_nsc)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn decode_quantizer_to_nsc(
        &self,
        tokens: &[u32],
        lengths: &[usize],
        batch_size: usize,
        codebooks: usize,
        frames: usize,
    ) -> AudioResult<Vec<f32>> {
        let context = self.decode_context()?;
        self.decode_quantizer_to_nsc_on_context(&context, tokens, lengths, batch_size, codebooks, frames)
    }

    fn decode_quantizer_to_nsc_array_on_context(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
        tokens: &[u32],
        lengths: &[usize],
        batch_size: usize,
        codebooks: usize,
        frames: usize,
        profile: &mut Option<AudioDecodeProfile>,
    ) -> AudioResult<Array<Metal>> {
        if codebooks != self.total_codebooks {
            return Err(AudioError::Runtime(format!(
                "FishAudio codebook mismatch: expected {}, got {codebooks}",
                self.total_codebooks
            )));
        }
        let expected_tokens = checked_product(&[batch_size, codebooks, frames])?;
        if tokens.len() != expected_tokens {
            return Err(AudioError::InvalidTokenShape {
                expected_tokens,
                actual_tokens: tokens.len(),
            });
        }
        if lengths.len() != batch_size {
            return Err(AudioError::InvalidTokenLengths {
                expected_lengths: batch_size,
                actual_lengths: lengths.len(),
            });
        }
        for &length in lengths {
            if length > frames {
                return Err(AudioError::InvalidTokenLengthValue {
                    length,
                    frames,
                });
            }
        }
        let quantizer_resources = self.quantizer_gpu_resources(context)?;
        if quantizer_resources.residual_quantizers + 1 != codebooks {
            return Err(AudioError::Runtime(format!(
                "FishAudio residual quantizer count mismatch: expected {}, got {}",
                codebooks.saturating_sub(1),
                quantizer_resources.residual_quantizers
            )));
        }

        let tokens_i32 = tokens
            .iter()
            .copied()
            .map(|token| {
                i32::try_from(token).map_err(|_| AudioError::Runtime(format!("token id out of i32 range: {token}")))
            })
            .collect::<AudioResult<Vec<_>>>()?;
        let lengths_i32 = convert_lengths_to_i32(lengths, frames)?;

        let mut tokens_array =
            context.create_array(&[batch_size, codebooks, frames], DataType::I32, "fishaudio_quantizer_tokens");
        tokens_array.as_slice_mut::<i32>().copy_from_slice(&tokens_i32);
        let mut lengths_array = context.create_array(&[batch_size], DataType::I32, "fishaudio_quantizer_lengths");
        lengths_array.as_slice_mut::<i32>().copy_from_slice(&lengths_i32);

        let output = context.create_array(
            &[batch_size, frames, self.input_dim],
            quantizer_resources.data_type,
            "fishaudio_quantizer_output_nsc",
        );
        let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
        let codebooks_i32 = usize_to_i32(codebooks, "codebooks")?;
        let frames_i32 = usize_to_i32(frames, "frames")?;
        let input_dim_i32 = usize_to_i32(self.input_dim, "input_dim")?;
        let codebook_dim_i32 = usize_to_i32(quantizer_resources.codebook_dim, "codebook_dim")?;
        let residual_quantizers_i32 = usize_to_i32(quantizer_resources.residual_quantizers, "residual_quantizers")?;
        let semantic_cardinality_i32 = usize_to_i32(quantizer_resources.semantic_cardinality, "semantic_cardinality")?;
        let residual_cardinality_i32 = usize_to_i32(quantizer_resources.residual_cardinality, "residual_cardinality")?;

        let encode_start = profile.is_some().then(Instant::now);
        let mut command_buffer = context
            .create_command_buffer()
            .map_err(|err| AudioError::Runtime(format!("failed to create quantizer command buffer: {err}")))?
            .start_encoding();
        let tokens_buffer = tokens_array.buffer();
        let tokens_buffer = tokens_buffer.borrow();
        let lengths_buffer = lengths_array.buffer();
        let lengths_buffer = lengths_buffer.borrow();
        let semantic_codebook_buffer = quantizer_resources.semantic_codebook.buffer();
        let semantic_codebook_buffer = semantic_codebook_buffer.borrow();
        let semantic_out_proj_buffer = quantizer_resources.semantic_out_proj.buffer();
        let semantic_out_proj_buffer = semantic_out_proj_buffer.borrow();
        let semantic_out_bias_buffer = quantizer_resources.semantic_out_bias.buffer();
        let semantic_out_bias_buffer = semantic_out_bias_buffer.borrow();
        let residual_codebooks_buffer = quantizer_resources.residual_codebooks.buffer();
        let residual_codebooks_buffer = residual_codebooks_buffer.borrow();
        let residual_out_proj_buffer = quantizer_resources.residual_out_proj.buffer();
        let residual_out_proj_buffer = residual_out_proj_buffer.borrow();
        let residual_out_bias_buffer = quantizer_resources.residual_out_bias.buffer();
        let residual_out_bias_buffer = residual_out_bias_buffer.borrow();
        let output_buffer = output.buffer();
        let mut output_buffer = output_buffer.borrow_mut();
        command_buffer.with_compute_encoder(|compute_encoder| {
            quantizer_resources.kernel.encode(
                &*tokens_buffer,
                &*lengths_buffer,
                &*semantic_codebook_buffer,
                &*semantic_out_proj_buffer,
                &*semantic_out_bias_buffer,
                &*residual_codebooks_buffer,
                &*residual_out_proj_buffer,
                &*residual_out_bias_buffer,
                &mut *output_buffer,
                batch_i32,
                codebooks_i32,
                frames_i32,
                input_dim_i32,
                codebook_dim_i32,
                residual_quantizers_i32,
                semantic_cardinality_i32,
                residual_cardinality_i32,
                compute_encoder,
            );
        });
        let cpu_encode_ms = encode_start.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
        let command_buffer = command_buffer.end_encoding().submit();
        let wait_start = profile.is_some().then(Instant::now);
        let command_buffer = command_buffer
            .wait_until_completed()
            .map_err(|err| AudioError::Runtime(format!("failed to wait for quantizer command buffer: {err}")))?;
        let cpu_wait_ms = wait_start.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
        push_audio_command_buffer_profile(profile, "quantizer", &command_buffer, cpu_encode_ms, cpu_wait_ms, None);

        Ok(output)
    }

    fn decode_quantizer_to_nsc_on_context(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
        tokens: &[u32],
        lengths: &[usize],
        batch_size: usize,
        codebooks: usize,
        frames: usize,
    ) -> AudioResult<Vec<f32>> {
        let mut profile = None;
        let output = self.decode_quantizer_to_nsc_array_on_context(
            context,
            tokens,
            lengths,
            batch_size,
            codebooks,
            frames,
            &mut profile,
        )?;
        Ok(read_array_to_f32_vec(&output)?)
    }

}
