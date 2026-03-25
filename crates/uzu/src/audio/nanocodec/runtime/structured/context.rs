use super::*;

impl StructuredAudioCodecGraph {
    fn conv1d_input_context<B: Backend>(layer: &StructuredAudioConv1d<B>) -> AudioResult<usize> {
        layer
            .kernel_size
            .checked_sub(1)
            .and_then(|value| value.checked_mul(layer.dilation))
            .ok_or(AudioError::Runtime("conv1d streaming context overflow".to_string()))
    }

    fn convtranspose_input_context<B: Backend>(
        output_context: usize,
        layer: &StructuredAudioConvTranspose1d<B>,
    ) -> AudioResult<usize> {
        let kernel_minus_one = layer.kernel_size.saturating_sub(1);
        let numerator = output_context
            .checked_add(kernel_minus_one)
            .ok_or(AudioError::Runtime("transpose streaming context overflow".to_string()))?;
        checked_div_ceil(numerator, layer.stride)
    }

    fn residual_unit_input_context<B: Backend>(
        output_context: usize,
        layer: &StructuredAudioResidualUnit<B>,
    ) -> AudioResult<usize> {
        let conv2 = Self::conv1d_input_context(&layer.conv2)?;
        let conv1 = Self::conv1d_input_context(&layer.conv1)?;
        output_context
            .checked_add(conv2)
            .and_then(|value| value.checked_add(conv1))
            .ok_or(AudioError::Runtime("residual-unit streaming context overflow".to_string()))
    }

    fn convnext_input_context<B: Backend>(
        output_context: usize,
        layer: &StructuredAudioConvNeXt<B>,
    ) -> AudioResult<usize> {
        output_context
            .checked_add(Self::conv1d_input_context(&layer.depthwise_conv)?)
            .ok_or(AudioError::Runtime("convnext streaming context overflow".to_string()))
    }

    fn decoder_block_input_context<B: Backend>(
        output_context: usize,
        layer: &StructuredAudioDecoderBlock<B>,
    ) -> AudioResult<usize> {
        let after_res3 = Self::residual_unit_input_context(output_context, &layer.res_unit3)?;
        let after_res2 = Self::residual_unit_input_context(after_res3, &layer.res_unit2)?;
        let after_res1 = Self::residual_unit_input_context(after_res2, &layer.res_unit1)?;
        Self::convtranspose_input_context(after_res1, &layer.trans_conv)
    }

    fn streaming_vocoder_context_frames<B: Backend>(
        &self,
        resources: &StructuredAudioRuntimeResources<B>,
    ) -> AudioResult<usize> {
        let vocoder = self.vocoder_graph(resources)?;
        let mut required = Self::conv1d_input_context(&vocoder.final_conv)?;

        for block in vocoder.decoder_blocks.iter().rev() {
            required = Self::decoder_block_input_context(required, block)?;
        }

        required = required
            .checked_add(Self::conv1d_input_context(&vocoder.first_conv)?)
            .ok_or(AudioError::Runtime("decoder-first streaming context overflow".to_string()))?;

        for (trans_conv, convnext) in vocoder.upsample_blocks.iter().rev() {
            required = Self::convnext_input_context(required, convnext)?;
            required = Self::convtranspose_input_context(required, trans_conv)?;
        }

        Ok(required)
    }

    fn post_module_streaming_context_frames(&self) -> Option<usize> {
        let mut context = 0usize;
        for layer in &self.config.quantizer_config.post_module_config.layer_configs {
            let window = layer.mixer_config.sliding_window_size()?;
            context = context.max(window.saturating_sub(1));
        }
        Some(context)
    }

    pub(in crate::audio::nanocodec::runtime) fn streaming_decode_context_frames<B: Backend>(
        &self,
        resources: &StructuredAudioRuntimeResources<B>,
    ) -> AudioResult<Option<usize>> {
        let vocoder_context = self.streaming_vocoder_context_frames(resources)?;
        let Some(post_module_context) = self.post_module_streaming_context_frames() else {
            return Ok(None);
        };
        Ok(Some(vocoder_context.max(post_module_context)))
    }

    pub(super) fn decode_quantizer_to_nsc_array_enqueued<B: Backend>(
        &self,
        resources: &StructuredAudioRuntimeResources<B>,
        context: &Rc<B::Context>,
        encoder: &mut Encoder<B>,
        tokens: &[u32],
        lengths: &[usize],
        batch_size: usize,
        codebooks: usize,
        frames: usize,
    ) -> AudioResult<Array<B>> {
        if codebooks != self.total_codebooks {
            return Err(AudioError::Runtime(format!(
                "structured audio codebook mismatch: expected {}, got {codebooks}",
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
        let quantizer_resources = self.quantizer_resources(resources)?;
        if quantizer_resources.residual_quantizers + 1 != codebooks {
            return Err(AudioError::Runtime(format!(
                "structured audio residual quantizer count mismatch: expected {}, got {}",
                codebooks.saturating_sub(1),
                quantizer_resources.residual_quantizers
            )));
        }

        let lengths_i32 = convert_lengths_to_i32(lengths, frames)?;

        let mut tokens_array = resources.token_staging(batch_size * codebooks * frames);
        tokens_array.as_slice_mut::<u32>()[..tokens.len()].copy_from_slice(tokens);
        let mut lengths_array = resources.length_staging(batch_size);
        lengths_array.as_slice_mut::<i32>()[..lengths_i32.len()].copy_from_slice(&lengths_i32);

        let output = context.create_array_zeros(
            &[batch_size, frames, self.input_dim],
            quantizer_resources.data_type,
            "structured_audio_quantizer_output_nsc",
        );
        let batch_i32 = usize_to_i32(batch_size, "batch_size")?;
        let codebooks_i32 = usize_to_i32(codebooks, "codebooks")?;
        let frames_i32 = usize_to_i32(frames, "frames")?;
        let input_dim_i32 = usize_to_i32(self.input_dim, "input_dim")?;
        let codebook_dim_i32 = usize_to_i32(quantizer_resources.codebook_dim, "codebook_dim")?;
        let residual_quantizers_i32 = usize_to_i32(quantizer_resources.residual_quantizers, "residual_quantizers")?;
        let semantic_cardinality_i32 = usize_to_i32(quantizer_resources.semantic_cardinality, "semantic_cardinality")?;
        let residual_cardinality_i32 = usize_to_i32(quantizer_resources.residual_cardinality, "residual_cardinality")?;

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
            encoder,
        );

        Ok(output)
    }
}
