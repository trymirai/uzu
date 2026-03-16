use super::*;

impl StructuredAudioCodecGraph {
    fn run_residual_unit_enqueued<B: Backend>(
        &self,
        context: &Rc<B::Context>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        input: &Array<B>,
        unit: &StructuredAudioResidualUnit<B>,
        lengths: &[i32],
        lengths_array: &Array<B>,
        batch_size: usize,
        channels: usize,
        seq_len: usize,
        kernels: &StructuredAudioKernelCache<B>,
    ) -> AudioResult<Array<B>> {
        let residual = input.clone();
        let x = snake1d_enqueue(
            context,
            command_buffer,
            input,
            &unit.snake1_alpha,
            batch_size,
            channels,
            seq_len,
            kernels,
        )?;
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
            kernels,
        )?;
        let x =
            snake1d_enqueue(context, command_buffer, &x, &unit.snake2_alpha, batch_size, channels, seq_len, kernels)?;
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
            kernels,
        )
    }

    pub(in crate::audio::nanocodec::runtime) fn submit_decode_padded<B: Backend>(
        &self,
        resources: &StructuredAudioRuntimeResources<B>,
        runtime_options: NanoCodecFsqRuntimeOptions,
        tokens: &[u32],
        lengths: &[usize],
        batch_size: usize,
        codebooks: usize,
        frames: usize,
    ) -> AudioResult<SubmittedDecodedPaddedAudio<B>> {
        if batch_size == 0 || frames == 0 {
            let out_lengths = lengths
                .iter()
                .map(|&length| {
                    length
                        .checked_mul(self.upsample_factor)
                        .ok_or(AudioError::Runtime("structured audio length scaling overflow".to_string()))
                })
                .collect::<AudioResult<Vec<_>>>()?;
            let context = resources.context().clone();
            return Ok(SubmittedDecodedPaddedAudio {
                output: context.create_array(&[0], DataType::F32, "structured_audio_empty_decode_output"),
                channels: 1,
                frames: out_lengths.iter().copied().max().unwrap_or(0),
                lengths: out_lengths,
                final_command_buffer: None,
            });
        }

        let mut lengths_i32 = lengths
            .iter()
            .map(|&length| {
                i32::try_from(length)
                    .map_err(|_| AudioError::Runtime("structured audio length exceeds i32 range".to_string()))
            })
            .collect::<AudioResult<Vec<_>>>()?;
        let context = resources.context().clone();
        let chunked_command_buffers = runtime_options.chunked_command_buffers;
        let micro_flush_min_elements = runtime_options.micro_flush_min_elements;
        let mut command_buffer = context
            .create_command_buffer()
            .map_err(|err| {
                AudioError::Runtime(format!("failed to create structured audio decode command buffer: {err}"))
            })?
            .start_encoding();

        let mut x;
        let mut x_layout = SequenceLayout::Nsc;
        let quantized_nsc = self.decode_quantizer_to_nsc_array_enqueued(
            resources,
            &context,
            &mut command_buffer,
            tokens,
            lengths,
            batch_size,
            codebooks,
            frames,
        )?;
        let flush_stage = |command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding| -> AudioResult<()> {
            if !chunked_command_buffers {
                return Ok(());
            }
            let next_command_buffer = context
                .create_command_buffer()
                .map_err(|err| {
                    AudioError::Runtime(format!("failed to create structured audio decode command buffer: {err}"))
                })?
                .start_encoding();
            std::mem::replace(command_buffer, next_command_buffer).end_encoding().submit();
            Ok(())
        };
        let _ = codebooks;
        flush_stage(&mut command_buffer)?;
        x = self.apply_post_module_gpu_on_array_enqueued(
            resources,
            &context,
            &mut command_buffer,
            &quantized_nsc,
            lengths,
            batch_size,
            frames,
        )?;
        flush_stage(&mut command_buffer)?;

        let vocoder_graph = self.vocoder_gpu_graph(resources)?;
        let kernels = resources.kernels(self.vocoder_data_type)?;
        let mut current_channels = self.input_dim;
        let mut current_frames = frames;
        let mut next_lengths_i32 = vec![0_i32; lengths_i32.len()];
        let mut lengths_array = context.create_array(&[lengths_i32.len()], DataType::I32, "structured_audio_lengths_a");
        write_i32_slice_into_array(&mut lengths_array, &lengths_i32)
            .map_err(|err| AudioError::Runtime(format!("structured_audio_lengths_a: {err}")))?;
        let mut next_lengths_array =
            context.create_array(&[lengths_i32.len()], DataType::I32, "structured_audio_lengths_b");

        for (block_index, (trans_conv, convnext)) in vocoder_graph.upsample_blocks.iter().enumerate() {
            if trans_conv.cin != current_channels {
                return Err(AudioError::Runtime(format!(
                    "structured audio upsampler input channel mismatch: expected {}, got {}",
                    trans_conv.cin, current_channels
                )));
            }
            let next_frames = current_frames
                .checked_mul(trans_conv.stride)
                .ok_or(AudioError::Runtime("structured audio upsampler frame overflow".to_string()))?;
            scale_lengths_i32_in_place(&lengths_i32, &mut next_lengths_i32, trans_conv.stride)?;
            write_i32_slice_into_array(&mut next_lengths_array, &next_lengths_i32)
                .map_err(|err| AudioError::Runtime(format!("structured_audio_upsample_lengths: {err}")))?;

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
                &kernels,
            )
            .map_err(|err| {
                AudioError::Runtime(format!(
                    "structured audio upsample block {block_index} transpose_conv failed: {err} (x_len={}, batch_size={}, cin={}, seq_len_in={}, seq_len_out={})",
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
                    &kernels,
                )
                .map_err(|err| {
                    AudioError::Runtime(format!("structured audio upsample block {block_index} convnext failed: {err}"))
                })?;
            flush_stage(&mut command_buffer)?;

            x_layout = SequenceLayout::Ncs;
            current_frames = next_frames;
            current_channels = trans_conv.cout;
            std::mem::swap(&mut lengths_i32, &mut next_lengths_i32);
            std::mem::swap(&mut lengths_array, &mut next_lengths_array);
        }

        if vocoder_graph.first_conv.cin != current_channels {
            return Err(AudioError::Runtime(format!(
                "structured audio decoder input channels mismatch: expected {}, got {}",
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
            &kernels,
        )?;
        current_channels = vocoder_graph.first_conv.cout;
        let _ = conv1d_estimated_macs(batch_size, current_frames, &vocoder_graph.first_conv)?;
        flush_stage(&mut command_buffer)?;

        for (_block_index, block) in vocoder_graph.decoder_blocks.iter().enumerate() {
            if block.trans_conv.cin != current_channels {
                return Err(AudioError::Runtime(format!(
                    "structured audio decoder block input mismatch: expected {}, got {}",
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
                &kernels,
            )?;

            let next_frames = current_frames
                .checked_mul(block.trans_conv.stride)
                .ok_or(AudioError::Runtime("structured audio decoder frame overflow".to_string()))?;
            scale_lengths_i32_in_place(&lengths_i32, &mut next_lengths_i32, block.trans_conv.stride)?;
            write_i32_slice_into_array(&mut next_lengths_array, &next_lengths_i32)
                .map_err(|err| AudioError::Runtime(format!("structured_audio_decoder_block_lengths: {err}")))?;

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
                &kernels,
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
                .ok_or(AudioError::Runtime("structured audio decoder element count overflow".to_string()))?;
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
                &kernels,
            )?;
            if chunked_command_buffers && active_elements >= micro_flush_min_elements {
                let _ = checked_add_usize(
                    trans_conv_estimated_macs,
                    res1_estimated_macs,
                    "decoder block res1 estimated MACs",
                )?;
                flush_stage(&mut command_buffer)?;
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
                &kernels,
            )?;
            if chunked_command_buffers && active_elements >= micro_flush_min_elements {
                flush_stage(&mut command_buffer)?;
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
                &kernels,
            )?;
            let _ = res3_estimated_macs;
            let _ = block_total_estimated_macs;
            flush_stage(&mut command_buffer)?;
        }

        x = snake1d_enqueue(
            &context,
            &mut command_buffer,
            &x,
            &vocoder_graph.final_snake_alpha,
            batch_size,
            current_channels,
            current_frames,
            &kernels,
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
            &kernels,
        )?;
        x = tanh_enqueue(&context, &mut command_buffer, &x, &kernels)?;

        let final_command_buffer = Some(command_buffer.end_encoding().submit());
        let out_lengths = lengths_i32
            .into_iter()
            .map(|length| {
                usize::try_from(length).map_err(|_| {
                    AudioError::Runtime("structured audio decoder produced invalid negative length".to_string())
                })
            })
            .collect::<AudioResult<Vec<_>>>()?;
        Ok(SubmittedDecodedPaddedAudio {
            output: x,
            channels: vocoder_graph.final_conv.cout,
            frames: current_frames,
            lengths: out_lengths,
            final_command_buffer,
        })
    }

    pub(in crate::audio::nanocodec::runtime) fn decode_padded<B: Backend>(
        &self,
        resources: &StructuredAudioRuntimeResources<B>,
        runtime_options: NanoCodecFsqRuntimeOptions,
        tokens: &[u32],
        lengths: &[usize],
        batch_size: usize,
        codebooks: usize,
        frames: usize,
    ) -> AudioResult<DecodedPaddedAudio> {
        self.submit_decode_padded(resources, runtime_options, tokens, lengths, batch_size, codebooks, frames)?.resolve()
    }
}
