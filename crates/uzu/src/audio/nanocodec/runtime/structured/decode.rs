impl StructuredAudioCodecGraph {
    fn run_residual_unit_enqueued(
        &self,
        context: &Rc<<Metal as Backend>::Context>,
        command_buffer: &mut MetalCommandBuffer,
        input: &Array<Metal>,
        unit: &StructuredAudioResidualUnitGpuLayer,
        lengths: &[i32],
        lengths_array: &Array<Metal>,
        batch_size: usize,
        channels: usize,
        seq_len: usize,
    ) -> AudioResult<Array<Metal>> {
        let residual = input.clone();
        let x = snake1d_enqueue(context, command_buffer, input, &unit.snake1_alpha, batch_size, channels, seq_len)?;
        let x = causal_conv1d_grouped_enqueue(
            context,
            command_buffer,
            &x,
            &unit.conv1,
            SequenceLayout::Ncs,
            lengths,
            lengths_array,
            batch_size,
            seq_len,
        )?;
        let x = snake1d_enqueue(context, command_buffer, &x, &unit.snake2_alpha, batch_size, channels, seq_len)?;
        causal_conv1d_grouped_residual_enqueue(
            context,
            command_buffer,
            &x,
            &residual,
            &unit.conv2,
            lengths,
            lengths_array,
            batch_size,
            seq_len,
        )
    }

    fn submit_decode_padded(
        &self,
        runtime_options: NanoCodecFsqRuntimeOptions,
        tokens: &[u32],
        lengths: &[usize],
        batch_size: usize,
        codebooks: usize,
        frames: usize,
    ) -> AudioResult<SubmittedDecodedPaddedAudio> {
        let collect_command_buffer_profile =
            runtime_options.collect_command_buffer_profile || runtime_options.capture_single_decode;
        if batch_size == 0 || frames == 0 {
            let out_lengths = lengths
                .iter()
                .map(|&length| {
                    length
                        .checked_mul(self.upsample_factor)
                        .ok_or(AudioError::Runtime("FishAudio length scaling overflow".to_string()))
                })
                .collect::<AudioResult<Vec<_>>>()?;
            let context = <Metal as Backend>::Context::new()
                .map_err(|err| AudioError::Runtime(format!("failed to create metal audio context: {err}")))?;
            return Ok(SubmittedDecodedPaddedAudio {
                output: context.create_array(&[0], DataType::F32, "fishaudio_empty_decode_output"),
                channels: 1,
                frames: out_lengths.iter().copied().max().unwrap_or(0),
                lengths: out_lengths,
                final_command_buffer: None,
                final_command_label: None,
                final_cpu_encode_ms: 0.0,
                decode_profile: None,
                capture: None,
            });
        }

        let mut lengths_i32 = lengths
            .iter()
            .map(|&length| {
                i32::try_from(length).map_err(|_| AudioError::Runtime("FishAudio length exceeds i32 range".to_string()))
            })
            .collect::<AudioResult<Vec<_>>>()?;
        let mut decode_profile = collect_command_buffer_profile.then(|| AudioDecodeProfile {
            batch_size,
            frames,
            codebooks,
            ..AudioDecodeProfile::default()
        });
        let capture = if runtime_options.capture_single_decode {
            Some(AudioCaptureGuard::start()?)
        } else {
            None
        };
        let context = if let Some(capture) = capture.as_ref() {
            capture.context()
        } else {
            self.decode_context()?
        };

        let can_use_gpu_latent_path = batch_size == 1 && lengths.first().copied() == Some(frames);
        let mut x;
        let mut x_layout = SequenceLayout::Nsc;
        let quantized_nsc = self.decode_quantizer_to_nsc_array_on_context(
            &context,
            tokens,
            lengths,
            batch_size,
            codebooks,
            frames,
            &mut decode_profile,
        )?;
        x = if can_use_gpu_latent_path {
            self.apply_post_module_gpu_on_array_single_batch(&context, &quantized_nsc, frames, &mut decode_profile)?
        } else {
            self.apply_post_module_gpu_on_array(
                &context,
                &quantized_nsc,
                lengths,
                batch_size,
                frames,
                &mut decode_profile,
            )?
        };

        let vocoder_graph = self.vocoder_gpu_graph(&context)?;
        let mut command_buffer = context
            .create_command_buffer()
            .map_err(|err| AudioError::Runtime(format!("failed to create FishAudio decode command buffer: {err}")))?
            .start_encoding();
        let profile_decoder_micro_stages = runtime_options.profile_decoder_micro_stages;
        let chunked_command_buffers = runtime_options.chunked_command_buffers;
        let micro_flush_min_elements = runtime_options.micro_flush_min_elements;
        let mut command_buffer_encode_start = decode_profile.is_some().then(Instant::now);
        let mut flush_stage = |label: String,
                               estimated_macs: Option<usize>,
                               command_buffer: &mut MetalCommandBuffer|
         -> AudioResult<()> {
            if !(chunked_command_buffers || profile_decoder_micro_stages) {
                return Ok(());
            }
            let cpu_encode_ms =
                command_buffer_encode_start.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
            let next_command_buffer = context
                .create_command_buffer()
                .map_err(|err| AudioError::Runtime(format!("failed to create FishAudio decode command buffer: {err}")))?
                .start_encoding();
            let submitted = std::mem::replace(command_buffer, next_command_buffer).end_encoding().submit();
            if decode_profile.is_some() {
                let wait_start = Instant::now();
                let completed = submitted.wait_until_completed().map_err(|err| {
                    AudioError::Runtime(format!("failed to wait for FishAudio decoder command buffer: {err}"))
                })?;
                let cpu_wait_ms = wait_start.elapsed().as_secs_f64() * 1000.0;
                push_audio_command_buffer_profile(
                    &mut decode_profile,
                    label,
                    &completed,
                    cpu_encode_ms,
                    cpu_wait_ms,
                    estimated_macs,
                );
            }
            command_buffer_encode_start = decode_profile.is_some().then(Instant::now);
            Ok(())
        };
        let mut current_channels = self.input_dim;
        let mut current_frames = frames;
        let mut next_lengths_i32 = vec![0_i32; lengths_i32.len()];
        let mut lengths_array = context.create_array(&[lengths_i32.len()], DataType::I32, "fishaudio_lengths_a");
        write_i32_slice_into_array(&mut lengths_array, &lengths_i32)
            .map_err(|err| AudioError::Runtime(format!("fishaudio_lengths_a: {err}")))?;
        let mut next_lengths_array = context.create_array(&[lengths_i32.len()], DataType::I32, "fishaudio_lengths_b");

        for (block_index, (trans_conv, convnext)) in vocoder_graph.upsample_blocks.iter().enumerate() {
            if trans_conv.cin != current_channels {
                return Err(AudioError::Runtime(format!(
                    "FishAudio upsampler input channel mismatch: expected {}, got {}",
                    trans_conv.cin, current_channels
                )));
            }
            let next_frames = current_frames
                .checked_mul(trans_conv.stride)
                .ok_or(AudioError::Runtime("FishAudio upsampler frame overflow".to_string()))?;
            scale_lengths_i32_in_place(&lengths_i32, &mut next_lengths_i32, trans_conv.stride)?;
            write_i32_slice_into_array(&mut next_lengths_array, &next_lengths_i32)
                .map_err(|err| AudioError::Runtime(format!("fishaudio_upsample_lengths: {err}")))?;

            x = causal_conv_transpose1d_causal_pad_enqueue(
                &context,
                &mut command_buffer,
                &x,
                trans_conv,
                &next_lengths_i32,
                batch_size,
                current_frames,
                next_frames,
                x_layout,
                &next_lengths_array,
            )
            .map_err(|err| {
                AudioError::Runtime(format!(
                    "FishAudio upsample block {block_index} transpose_conv failed: {err} (x_len={}, batch_size={}, cin={}, seq_len_in={}, seq_len_out={})",
                    x.num_elements(),
                    batch_size,
                    trans_conv.cin,
                    current_frames,
                    next_frames
                ))
            })?;
            x = self
                .apply_convnext_ncs_enqueued(
                    &context,
                    &mut command_buffer,
                    &x,
                    convnext,
                    &next_lengths_i32,
                    &next_lengths_array,
                    batch_size,
                    trans_conv.cout,
                    next_frames,
                )
                .map_err(|err| {
                    AudioError::Runtime(format!("FishAudio upsample block {block_index} convnext failed: {err}"))
                })?;
            flush_stage(format!("upsample_block_{block_index}"), None, &mut command_buffer)?;

            x_layout = SequenceLayout::Ncs;
            current_frames = next_frames;
            current_channels = trans_conv.cout;
            std::mem::swap(&mut lengths_i32, &mut next_lengths_i32);
            std::mem::swap(&mut lengths_array, &mut next_lengths_array);
        }

        if vocoder_graph.first_conv.cin != current_channels {
            return Err(AudioError::Runtime(format!(
                "FishAudio decoder input channels mismatch: expected {}, got {}",
                vocoder_graph.first_conv.cin, current_channels
            )));
        }
        x = causal_conv1d_grouped_enqueue(
            &context,
            &mut command_buffer,
            &x,
            &vocoder_graph.first_conv,
            x_layout,
            &lengths_i32,
            &lengths_array,
            batch_size,
            current_frames,
        )?;
        current_channels = vocoder_graph.first_conv.cout;
        flush_stage(
            "decoder_first_conv".to_string(),
            Some(conv1d_estimated_macs(batch_size, current_frames, &vocoder_graph.first_conv)?),
            &mut command_buffer,
        )?;

        for (block_index, block) in vocoder_graph.decoder_blocks.iter().enumerate() {
            if block.trans_conv.cin != current_channels {
                return Err(AudioError::Runtime(format!(
                    "FishAudio decoder block input mismatch: expected {}, got {}",
                    block.trans_conv.cin, current_channels
                )));
            }
            x = snake1d_enqueue(
                &context,
                &mut command_buffer,
                &x,
                &block.snake_alpha,
                batch_size,
                current_channels,
                current_frames,
            )?;

            let next_frames = current_frames
                .checked_mul(block.trans_conv.stride)
                .ok_or(AudioError::Runtime("FishAudio decoder frame overflow".to_string()))?;
            scale_lengths_i32_in_place(&lengths_i32, &mut next_lengths_i32, block.trans_conv.stride)?;
            write_i32_slice_into_array(&mut next_lengths_array, &next_lengths_i32)
                .map_err(|err| AudioError::Runtime(format!("fishaudio_decoder_block_lengths: {err}")))?;

            x = causal_conv_transpose1d_causal_pad_enqueue(
                &context,
                &mut command_buffer,
                &x,
                &block.trans_conv,
                &next_lengths_i32,
                batch_size,
                current_frames,
                next_frames,
                SequenceLayout::Ncs,
                &next_lengths_array,
            )?;
            let trans_conv_estimated_macs =
                convtranspose_estimated_macs(batch_size, current_frames, &block.trans_conv)?;

            current_frames = next_frames;
            current_channels = block.trans_conv.cout;
            std::mem::swap(&mut lengths_i32, &mut next_lengths_i32);
            std::mem::swap(&mut lengths_array, &mut next_lengths_array);
            let active_elements = batch_size
                .checked_mul(current_channels)
                .and_then(|value| value.checked_mul(current_frames))
                .ok_or(AudioError::Runtime("FishAudio decoder element count overflow".to_string()))?;
            let res1_estimated_macs = residual_unit_estimated_macs(batch_size, current_frames, &block.res_unit1)?;
            let res2_estimated_macs = residual_unit_estimated_macs(batch_size, current_frames, &block.res_unit2)?;
            let res3_estimated_macs = residual_unit_estimated_macs(batch_size, current_frames, &block.res_unit3)?;
            let block_total_estimated_macs = checked_add_usize(
                checked_add_usize(
                    checked_add_usize(trans_conv_estimated_macs, res1_estimated_macs, "decoder block estimated MACs")?,
                    res2_estimated_macs,
                    "decoder block estimated MACs",
                )?,
                res3_estimated_macs,
                "decoder block estimated MACs",
            )?;

            if profile_decoder_micro_stages {
                flush_stage(
                    format!("decoder_block_{block_index}_trans_conv"),
                    Some(trans_conv_estimated_macs),
                    &mut command_buffer,
                )?;
            }

            x = self.run_residual_unit_enqueued(
                &context,
                &mut command_buffer,
                &x,
                &block.res_unit1,
                &lengths_i32,
                &lengths_array,
                batch_size,
                current_channels,
                current_frames,
            )?;
            if profile_decoder_micro_stages {
                flush_stage(
                    format!("decoder_block_{block_index}_res1"),
                    Some(res1_estimated_macs),
                    &mut command_buffer,
                )?;
            } else if chunked_command_buffers && active_elements >= micro_flush_min_elements {
                flush_stage(
                    format!("decoder_block_{block_index}_res1"),
                    Some(checked_add_usize(
                        trans_conv_estimated_macs,
                        res1_estimated_macs,
                        "decoder block res1 estimated MACs",
                    )?),
                    &mut command_buffer,
                )?;
            }
            x = self.run_residual_unit_enqueued(
                &context,
                &mut command_buffer,
                &x,
                &block.res_unit2,
                &lengths_i32,
                &lengths_array,
                batch_size,
                current_channels,
                current_frames,
            )?;
            if profile_decoder_micro_stages || (chunked_command_buffers && active_elements >= micro_flush_min_elements)
            {
                flush_stage(
                    format!("decoder_block_{block_index}_res2"),
                    Some(res2_estimated_macs),
                    &mut command_buffer,
                )?;
            }
            x = self.run_residual_unit_enqueued(
                &context,
                &mut command_buffer,
                &x,
                &block.res_unit3,
                &lengths_i32,
                &lengths_array,
                batch_size,
                current_channels,
                current_frames,
            )?;
            flush_stage(
                if profile_decoder_micro_stages {
                    format!("decoder_block_{block_index}_res3")
                } else {
                    format!("decoder_block_{block_index}")
                },
                Some(if profile_decoder_micro_stages {
                    res3_estimated_macs
                } else {
                    block_total_estimated_macs
                }),
                &mut command_buffer,
            )?;
        }

        x = snake1d_enqueue(
            &context,
            &mut command_buffer,
            &x,
            &vocoder_graph.final_snake_alpha,
            batch_size,
            current_channels,
            current_frames,
        )?;
        x = causal_conv1d_grouped_enqueue(
            &context,
            &mut command_buffer,
            &x,
            &vocoder_graph.final_conv,
            SequenceLayout::Ncs,
            &lengths_i32,
            &lengths_array,
            batch_size,
            current_frames,
        )?;
        x = tanh_enqueue(&context, &mut command_buffer, &x)?;

        let final_cpu_encode_ms =
            command_buffer_encode_start.map(|start| start.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
        let final_command_label = Some("decoder_final".to_string());
        let final_command_buffer = Some(command_buffer.end_encoding().submit());
        let out_lengths = lengths_i32
            .into_iter()
            .map(|length| {
                usize::try_from(length)
                    .map_err(|_| AudioError::Runtime("FishAudio decoder produced invalid negative length".to_string()))
            })
            .collect::<AudioResult<Vec<_>>>()?;
        Ok(SubmittedDecodedPaddedAudio {
            output: x,
            channels: vocoder_graph.final_conv.cout,
            frames: current_frames,
            lengths: out_lengths,
            final_command_buffer,
            final_command_label,
            final_cpu_encode_ms,
            decode_profile,
            capture,
        })
    }

    fn decode_padded(
        &self,
        runtime_options: NanoCodecFsqRuntimeOptions,
        tokens: &[u32],
        lengths: &[usize],
        batch_size: usize,
        codebooks: usize,
        frames: usize,
    ) -> AudioResult<(DecodedPaddedAudio, Option<AudioDecodeProfile>)> {
        self.submit_decode_padded(runtime_options, tokens, lengths, batch_size, codebooks, frames)?.resolve()
    }
}
