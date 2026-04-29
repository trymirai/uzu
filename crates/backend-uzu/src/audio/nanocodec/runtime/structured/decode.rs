use std::sync::mpsc::channel;

use super::*;
use crate::{array::Array, audio::nanocodec::runtime::profile::SubmittedDecodedPaddedAudio};

impl StructuredAudioCodecGraph {
    fn run_residual_unit_enqueued<B: Backend>(
        &self,
        resources: &StructuredAudioRuntimeResources<B>,
        encoder: &mut Encoder<B>,
        input: &Array<B>,
        unit: &StructuredAudioResidualUnit<B>,
        lengths: &[i32],
        lengths_array: &Array<B>,
        batch_size: usize,
        channels: usize,
        seq_len: usize,
        kernels: &StructuredAudioKernelCache<B>,
    ) -> AudioResult<Array<B>> {
        let ws = &resources.decode_workspace;
        let res_snake1 = ws.next_scratch(encoder, &[batch_size, channels, seq_len], self.vocoder_data_type);
        let x = snake1d_enqueue(
            encoder,
            input,
            res_snake1,
            &unit.snake1_alpha,
            batch_size,
            channels,
            seq_len,
            self.vocoder_data_type,
            kernels,
        )?;
        let res_conv1 = ws.next_scratch(encoder, &[batch_size, channels, seq_len], self.vocoder_data_type);
        let x = causal_conv1d_grouped_enqueue(
            encoder,
            &x,
            res_conv1,
            &unit.conv1,
            SequenceLayout::Ncs,
            lengths,
            lengths_array,
            batch_size,
            seq_len,
            self.vocoder_data_type,
            kernels,
        )?;
        let res_snake2 = ws.next_scratch(encoder, &[batch_size, channels, seq_len], self.vocoder_data_type);
        let x = snake1d_enqueue(
            encoder,
            &x,
            res_snake2,
            &unit.snake2_alpha,
            batch_size,
            channels,
            seq_len,
            self.vocoder_data_type,
            kernels,
        )?;
        let res_conv2 = ws.next_scratch(encoder, &[batch_size, channels, seq_len], self.vocoder_data_type);
        causal_conv1d_grouped_residual_enqueue(
            encoder,
            &x,
            input,
            res_conv2,
            &unit.conv2,
            lengths,
            lengths_array,
            batch_size,
            seq_len,
            self.vocoder_data_type,
            kernels,
        )
    }

    pub(in crate::audio::nanocodec::runtime) fn submit_decode_padded<B: Backend>(
        &self,
        resources: &StructuredAudioRuntimeResources<B>,
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
            return Ok(SubmittedDecodedPaddedAudio::Ready(DecodedPaddedAudio {
                samples: Vec::new(),
                channels: 1,
                frames: out_lengths.iter().copied().max().unwrap_or(0),
                lengths: out_lengths,
            }));
        }

        let mut lengths_i32 = lengths
            .iter()
            .map(|&length| {
                i32::try_from(length)
                    .map_err(|_| AudioError::Runtime("structured audio length exceeds i32 range".to_string()))
            })
            .collect::<AudioResult<Vec<_>>>()?;
        let context = resources.context().clone();
        let mut encoder = Encoder::new(context.as_ref()).map_err(|err| {
            AudioError::Runtime(format!("failed to create structured audio decode command buffer: {err}"))
        })?;

        let ws = &resources.decode_workspace;
        let vocoder_graph = self.vocoder_graph(resources)?;
        let kernels = resources.kernels(self.vocoder_data_type)?;

        let mut x;
        let mut x_layout = SequenceLayout::Nsc;
        let quantized_nsc = self.decode_quantizer_to_nsc_array_enqueued(
            resources,
            &mut encoder,
            tokens,
            lengths,
            batch_size,
            codebooks,
            frames,
        )?;
        let retained_inputs = quantized_nsc.retained_inputs;
        x = self.apply_post_module_enqueued(
            resources,
            &mut encoder,
            quantized_nsc.output,
            lengths,
            batch_size,
            frames,
        )?;

        let mut current_channels = self.input_dim;
        let mut current_frames = frames;
        let mut next_lengths_i32 = vec![0_i32; lengths_i32.len()];
        let mut lengths_array = ws.lengths_array(&mut encoder, lengths_i32.len());
        lengths_array.as_slice_mut::<i32>().copy_from_slice(&lengths_i32);
        let mut next_lengths_array = ws.lengths_array(&mut encoder, lengths_i32.len());

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
            next_lengths_array.as_slice_mut::<i32>().copy_from_slice(&next_lengths_i32);

            let up_tconv =
                ws.next_scratch(&mut encoder, &[batch_size, trans_conv.cout, next_frames], self.vocoder_data_type);
            x = causal_conv_transpose1d_causal_pad_enqueue(
                &mut encoder,
                &x,
                up_tconv,
                trans_conv,
                &next_lengths_i32,
                batch_size,
                current_frames,
                next_frames,
                x_layout,
                &next_lengths_array,
                self.vocoder_data_type,
                &kernels,
            )
            .map_err(|err| {
                AudioError::Runtime(format!(
                    "structured audio upsample block {block_index} transpose_conv failed: {err} (x_len={}, batch_size={}, cin={}, seq_len_in={}, seq_len_out={})",
                    x.size(),
                    batch_size,
                    trans_conv.cin,
                    current_frames,
                    next_frames
                ))
            })?;
            x = self
                .apply_convnext_ncs_enqueued(
                    resources,
                    &mut encoder,
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
        let dec_first_conv = ws.next_scratch(
            &mut encoder,
            &[batch_size, vocoder_graph.first_conv.cout, current_frames],
            self.vocoder_data_type,
        );
        x = causal_conv1d_grouped_enqueue(
            &mut encoder,
            &x,
            dec_first_conv,
            &vocoder_graph.first_conv,
            x_layout,
            &lengths_i32,
            &lengths_array,
            batch_size,
            current_frames,
            self.vocoder_data_type,
            &kernels,
        )?;
        current_channels = vocoder_graph.first_conv.cout;

        for block in vocoder_graph.decoder_blocks.iter() {
            if block.trans_conv.cin != current_channels {
                return Err(AudioError::Runtime(format!(
                    "structured audio decoder block input mismatch: expected {}, got {}",
                    block.trans_conv.cin, current_channels
                )));
            }
            let dec_snake =
                ws.next_scratch(&mut encoder, &[batch_size, current_channels, current_frames], self.vocoder_data_type);
            x = snake1d_enqueue(
                &mut encoder,
                &x,
                dec_snake,
                &block.snake_alpha,
                batch_size,
                current_channels,
                current_frames,
                self.vocoder_data_type,
                &kernels,
            )?;

            let next_frames = current_frames
                .checked_mul(block.trans_conv.stride)
                .ok_or(AudioError::Runtime("structured audio decoder frame overflow".to_string()))?;
            scale_lengths_i32_in_place(&lengths_i32, &mut next_lengths_i32, block.trans_conv.stride)?;
            next_lengths_array.as_slice_mut::<i32>().copy_from_slice(&next_lengths_i32);

            let dec_tconv = ws.next_scratch(
                &mut encoder,
                &[batch_size, block.trans_conv.cout, next_frames],
                self.vocoder_data_type,
            );
            x = causal_conv_transpose1d_causal_pad_enqueue(
                &mut encoder,
                &x,
                dec_tconv,
                &block.trans_conv,
                &next_lengths_i32,
                batch_size,
                current_frames,
                next_frames,
                SequenceLayout::Ncs,
                &next_lengths_array,
                self.vocoder_data_type,
                &kernels,
            )?;

            current_frames = next_frames;
            current_channels = block.trans_conv.cout;
            std::mem::swap(&mut lengths_i32, &mut next_lengths_i32);
            std::mem::swap(&mut lengths_array, &mut next_lengths_array);

            x = self.run_residual_unit_enqueued(
                resources,
                &mut encoder,
                &x,
                &block.res_unit1,
                &lengths_i32,
                &lengths_array,
                batch_size,
                current_channels,
                current_frames,
                &kernels,
            )?;
            x = self.run_residual_unit_enqueued(
                resources,
                &mut encoder,
                &x,
                &block.res_unit2,
                &lengths_i32,
                &lengths_array,
                batch_size,
                current_channels,
                current_frames,
                &kernels,
            )?;
            x = self.run_residual_unit_enqueued(
                resources,
                &mut encoder,
                &x,
                &block.res_unit3,
                &lengths_i32,
                &lengths_array,
                batch_size,
                current_channels,
                current_frames,
                &kernels,
            )?;
        }

        let final_snake =
            ws.next_scratch(&mut encoder, &[batch_size, current_channels, current_frames], self.vocoder_data_type);
        x = snake1d_enqueue(
            &mut encoder,
            &x,
            final_snake,
            &vocoder_graph.final_snake_alpha,
            batch_size,
            current_channels,
            current_frames,
            self.vocoder_data_type,
            &kernels,
        )?;
        let final_conv = ws.next_scratch(
            &mut encoder,
            &[batch_size, vocoder_graph.final_conv.cout, current_frames],
            self.vocoder_data_type,
        );
        x = causal_conv1d_grouped_enqueue(
            &mut encoder,
            &x,
            final_conv,
            &vocoder_graph.final_conv,
            SequenceLayout::Ncs,
            &lengths_i32,
            &lengths_array,
            batch_size,
            current_frames,
            self.vocoder_data_type,
            &kernels,
        )?;
        let tanh_output = ws.next_scratch(
            &mut encoder,
            &[batch_size, vocoder_graph.final_conv.cout, current_frames],
            self.vocoder_data_type,
        );
        x = tanh_enqueue(&mut encoder, &x, tanh_output, &kernels)?;

        let (completion_sender, completion_notification) = channel();
        encoder.add_completion_handler({
            move |_| {
                let _ = completion_sender.send(());
            }
        });
        let out_lengths = lengths_i32
            .into_iter()
            .map(|length| {
                usize::try_from(length).map_err(|_| {
                    AudioError::Runtime("structured audio decoder produced invalid negative length".to_string())
                })
            })
            .collect::<AudioResult<Vec<_>>>()?;
        let final_command_buffer = encoder.end_encoding().submit();
        Ok(SubmittedDecodedPaddedAudio::Pending {
            output: x.into_allocation(),
            retained_inputs,
            data_type: self.vocoder_data_type,
            channels: vocoder_graph.final_conv.cout,
            frames: current_frames,
            lengths: out_lengths,
            final_command_buffer,
            completion_notification,
        })
    }

    pub(in crate::audio::nanocodec::runtime) fn decode_padded<B: Backend>(
        &self,
        resources: &StructuredAudioRuntimeResources<B>,
        tokens: &[u32],
        lengths: &[usize],
        batch_size: usize,
        codebooks: usize,
        frames: usize,
    ) -> AudioResult<DecodedPaddedAudio> {
        self.submit_decode_padded(resources, tokens, lengths, batch_size, codebooks, frames)?.resolve()
    }
}
