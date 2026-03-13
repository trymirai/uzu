use super::{
    AdaptiveChunkController, AudioDecodeStepStats, DEFAULT_CHUNK_EMA_ALPHA, DEFAULT_CHUNK_HYSTERESIS_FRACTION,
    DEFAULT_STUB_SEED, DEFAULT_TTS_RANDOM_SEED, MatrixF32, PendingAudioChunkBackend, PendingStreamingChunk,
    StreamingTokenAccumulator, TextDecoderFollowupStrategy, TextSamplingState, audio_decode_streaming_mode,
    build_semantic_sampling_mask_row, clear_token_in_sampling_mask, expand_token_mask_for_sampling_row,
    generate_stub_tokens, load_stub_seed, maybe_flush_pending_stream_chunk, normalize_rendered_prompt,
    semantic_token_to_code,
};
use crate::{
    DataType,
    array::ArrayContextExt,
    audio::{AudioPcmBatch, AudioTokenPacking},
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending,
            Context, Kernels,
            kernel::RepetitionPenaltyKernel,
        },
        metal::Metal,
    },
};
use crate::config::{TtsMessageProcessorConfig, TtsModelConfig};
use crate::session::config::{
    TextDecoderRuntimeConfig, TextSamplingConfig, TtsChunkPolicy, TtsRunConfig, TtsVocoderStreamingMode,
};
use crate::session::parameter::{ConfigResolvableValue, SamplingMethod};
use crate::session::tts_session::AudioDecodeStreamingMode;
use crate::session::types::{Error, Message};

#[test]
fn missing_seed_file_uses_default_path() {
    let seed = load_stub_seed("/does/not/exist/model.safetensors".into()).unwrap_or(DEFAULT_STUB_SEED);
    assert_eq!(seed, DEFAULT_STUB_SEED);
}

#[test]
fn stub_token_generation_is_seeded_and_bounded() {
    let a = generate_stub_tokens(3, 8, 17, 123);
    let b = generate_stub_tokens(3, 8, 17, 123);
    let c = generate_stub_tokens(3, 8, 17, 456);

    assert_eq!(a, b);
    assert_ne!(a, c);
    assert!(a.iter().all(|&value| value < 17));
}

#[test]
fn semantic_token_to_code_respects_bounds() {
    assert_eq!(semantic_token_to_code(5, 5, 9, 4), 0);
    assert_eq!(semantic_token_to_code(7, 5, 9, 4), 2);
    assert_eq!(semantic_token_to_code(9, 5, 9, 4), 3);
    assert_eq!(semantic_token_to_code(11, 5, 9, 4), 0);
    assert_eq!(semantic_token_to_code(7, 10, 9, 4), 0);
    assert_eq!(semantic_token_to_code(7, 5, 9, 0), 0);
}

#[test]
fn semantic_sampling_mask_row_includes_band_and_im_end() {
    let mask = build_semantic_sampling_mask_row(96, 64, 79, 12).expect("mask");
    let bit = |token: usize| -> bool {
        let word = token / 32;
        let offset = token % 32;
        ((mask[word] >> offset) & 1) == 1
    };

    assert!(bit(12), "im_end must be selectable");
    assert!(bit(64) && bit(79), "semantic range endpoints must be selectable");
    assert!(!bit(63), "tokens below semantic range must be masked");
    assert!(!bit(80), "tokens above semantic range must be masked");
}

#[test]
fn rendered_prompt_drops_template_newlines() {
    let template = "\n{% for message in messages %}<|{{message.style}}|><|{{message.speaker_id}}|>{{message.content}}{% endfor %}\n";
    let rendered = "\n<|interleave|><|speaker:0|>hello\n".to_string();
    let normalized = normalize_rendered_prompt(rendered, template, true);
    assert_eq!(normalized, "<|interleave|><|speaker:0|>hello");
}

#[test]
fn prompt_message_context_uses_config_defaults_and_preserves_role() {
    let message_processor_config = TtsMessageProcessorConfig {
        prompt_template: String::from("{{messages[0].content}}"),
        drop_initial_newline: true,
        system_role_name: String::from("system"),
        user_role_name: String::from("user"),
        assistant_role_name: String::from("assistant"),
        default_message_fields: std::collections::BTreeMap::from([
            (String::from("speaker_id"), String::from("speaker:42")),
            (String::from("style"), String::from("relaxed")),
        ]),
    };
    let rendered =
        Message::assistant("hello".to_string(), Some("thinking".to_string())).resolve(&message_processor_config);

    assert_eq!(rendered.get("content"), Some(&String::from("hello")));
    assert_eq!(rendered.get("role"), Some(&String::from("assistant")));
    assert_eq!(rendered.get("speaker_id"), Some(&String::from("speaker:42")));
    assert_eq!(rendered.get("style"), Some(&String::from("relaxed")));
    assert_eq!(rendered.get("reasoning_content"), Some(&String::from("thinking")));
}

#[test]
fn stub_text_decoder_backend_rejects_cardinality_mismatch() {
    let model_config: TtsModelConfig = serde_json::from_value(serde_json::json!({
        "tts_config": {
            "text_decoder_config": {
                "type": "StubTextDecoderConfig",
                "num_codebooks": 2,
                "codebook_size": 49
            },
            "audio_decoder_config": {
                "type": "NanoCodecConfig",
                "samplerate": 24000,
                "quantizer_config": {
                    "num_groups": 2,
                    "quantizer_config": {
                        "num_levels": [8, 6]
                    }
                },
                "decoder_config": {
                    "activation_config": {
                        "leaky_relu_negative_slope": 0.01
                    }
                },
                "base_channels": 32,
                "up_sample_rates": [2, 2],
                "resblock_kernel_sizes": [3],
                "resblock_dilations": [1]
            },
            "vocoder_config": {}
        },
        "message_processor_config": {
            "prompt_template": "{{messages[0].content}}"
        }
    }))
    .expect("tts model config");

    let audio = model_config.create_audio_generation_context().expect("audio context");
    let result = super::backend_factory::build_text_decoder_backend(
        &model_config,
        &audio,
        std::path::Path::new("."),
        &crate::session::config::TtsSessionOptions::default(),
    );

    assert!(result.is_err(), "stub backend should reject codebook cardinality mismatches");
}

#[test]
fn expanded_sampling_mask_targets_only_sampling_row() {
    let row_mask = vec![0xFFFF_0000u32, 0x0000_FFFFu32];
    let expanded = expand_token_mask_for_sampling_row(&row_mask, 3).expect("expanded");
    assert_eq!(expanded.len(), 6);
    assert_eq!(&expanded[0..2], &[u32::MAX, u32::MAX]);
    assert_eq!(&expanded[2..4], &[u32::MAX, u32::MAX]);
    assert_eq!(&expanded[4..6], row_mask.as_slice());
}

#[test]
fn text_decoder_defaults_are_stable() {
    let config = TextDecoderRuntimeConfig::default();
    assert_eq!(DEFAULT_TTS_RANDOM_SEED, 123);
    assert_eq!(config.min_frames_before_im_end, 8);
    assert_eq!(config.prefill_step_size, 128);
    assert_eq!(config.followup_strategy, TextDecoderFollowupStrategy::AsyncChain);
    assert_eq!(config.sampling.temperature, 0.8008);
    assert_eq!(config.sampling.top_p, 0.8008);
    assert_eq!(config.sampling.repetition_penalty, 1.1);
}

#[test]
fn clearing_token_from_sampling_mask_removes_bit() {
    let mut mask = build_semantic_sampling_mask_row(96, 64, 79, 12).expect("mask").into_vec();
    clear_token_in_sampling_mask(&mut mask, 12).expect("clear");
    let word = 12 / 32;
    let bit = 12 % 32;
    assert_eq!((mask[word] >> bit) & 1, 0);
}

#[test]
fn fishaudio_sampling_seed_stream_is_deterministic() {
    let config = TextSamplingConfig::default();
    let mut a = TextSamplingState::from_config(123, &config);
    let mut b = TextSamplingState::from_config(123, &config);
    let seeds_a = (0..8).map(|_| a.next_seed()).collect::<Vec<_>>();
    let seeds_b = (0..8).map(|_| b.next_seed()).collect::<Vec<_>>();
    assert_eq!(seeds_a, seeds_b);
}

#[test]
fn fishaudio_sampling_top_p_zero_switches_to_greedy() {
    let sampler = TextSamplingState::with_params(999, 0.8, 0.0, 1.1);
    assert_eq!(sampler.method(), SamplingMethod::Greedy);
}

#[test]
fn fishaudio_sampling_positive_top_p_uses_stochastic_mode() {
    let sampler = TextSamplingState::with_params(999, 0.8, 0.8, 1.1);
    assert!(matches!(sampler.method(), SamplingMethod::AdvancedStochastic { .. }));
}

#[test]
fn repetition_penalty_kernel_adjusts_selected_logits() {
    let context = <Metal as Backend>::Context::new().expect("MetalContext");
    let kernel = <<Metal as Backend>::Kernels as Kernels>::RepetitionPenaltyKernel::new(&context, DataType::F32)
        .expect("repetition penalty kernel");

    let batch_size = 2usize;
    let vocab_size = 8usize;
    let max_previous_tokens = 4usize;
    let penalty = 1.25f32;

    let original_logits = vec![
        0.5, -1.0, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0, //
        8.0, -9.0, 10.0, 11.0, -12.0, 13.0, 14.0, -15.0,
    ];
    let mut logits = context.create_array(&[batch_size, vocab_size], DataType::F32, "repetition_penalty_logits");
    logits.as_slice_mut::<f32>().copy_from_slice(&original_logits);

    let previous_tokens_values: [u32; 8] = [1, 4, 0, 0, 0, 7, 0, 0];
    let mut previous_tokens =
        context.create_array(&[batch_size, max_previous_tokens], DataType::U32, "repetition_penalty_tokens");
    previous_tokens.as_slice_mut::<u32>().copy_from_slice(&previous_tokens_values);

    let previous_counts_values: [u32; 2] = [2, 2];
    let mut previous_counts =
        context.create_array(&[batch_size], DataType::U32, "repetition_penalty_counts");
    previous_counts.as_slice_mut::<u32>().copy_from_slice(&previous_counts_values);

    let mut command_buffer = context.create_command_buffer().expect("command buffer").start_encoding();
    {
        let logits_buffer = logits.buffer();
        let mut logits_buffer = logits_buffer.borrow_mut();
        let previous_tokens_buffer = previous_tokens.buffer();
        let previous_tokens_buffer = previous_tokens_buffer.borrow();
        let previous_counts_buffer = previous_counts.buffer();
        let previous_counts_buffer = previous_counts_buffer.borrow();
        kernel.encode(
            &mut *logits_buffer,
            &*previous_tokens_buffer,
            &*previous_counts_buffer,
            batch_size as u32,
            vocab_size as u32,
            max_previous_tokens as u32,
            penalty,
            &mut command_buffer,
        );
    }
    command_buffer
        .end_encoding()
        .submit()
        .wait_until_completed()
        .expect("command buffer completed");

    let adjusted = logits.as_slice::<f32>();
    let expected = vec![
        0.5, -1.25, 2.0, -3.0, 3.2, 5.0, -6.0, 7.0, //
        6.4, -9.0, 10.0, 11.0, -12.0, 13.0, 14.0, -18.75,
    ];
    for (index, (&got, &exp)) in adjusted.iter().zip(expected.iter()).enumerate() {
        let delta = (got - exp).abs();
        assert!(delta <= 1e-5, "index={index}: expected {exp}, got {got}, delta={delta}");
    }
}

#[test]
fn matrix_f32_row_and_matmul_work() {
    let matrix = MatrixF32 {
        rows: 2,
        cols: 3,
        values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    };

    assert_eq!(matrix.row(0), Some([1.0_f32, 2.0, 3.0].as_slice()));
    assert_eq!(matrix.row(1), Some([4.0_f32, 5.0, 6.0].as_slice()));
    assert_eq!(matrix.row(2), None);
    let mut out = [0.0_f32; 2];
    assert_eq!(matrix.matmul_into(&[1.0, 0.0, 1.0], &mut out), Some(()));
    assert_eq!(out, [4.0, 10.0]);
    assert_eq!(matrix.matmul_into(&[1.0, 0.0], &mut out), None);
}

#[test]
fn streaming_token_accumulator_builds_codebook_major_grid() {
    let mut accumulator = StreamingTokenAccumulator::new(3).expect("accumulator");
    accumulator.push_frame(&[1, 10, 100]).expect("frame 0");
    accumulator.push_frame(&[2, 20, 200]).expect("frame 1");

    let grid = accumulator.to_grid().expect("grid");
    assert_eq!(grid.batch_size(), 1);
    assert_eq!(grid.codebooks(), 3);
    assert_eq!(grid.frames(), 2);
    assert_eq!(grid.packing(), AudioTokenPacking::CodebookMajor);
    assert_eq!(grid.tokens(), &[1, 2, 10, 20, 100, 200]);
}

#[test]
fn streaming_token_accumulator_rejects_wrong_frame_width() {
    let mut accumulator = StreamingTokenAccumulator::new(2).expect("accumulator");
    let error = accumulator.push_frame(&[1]).expect_err("must reject wrong width");
    assert!(matches!(error, crate::session::types::Error::GenerateFailed));
}

#[test]
fn adaptive_chunk_controller_applies_hysteresis() {
    let config = TtsRunConfig {
        min_chunk_frames: 16,
        max_chunk_frames: 256,
        target_emit_latency_ms: 80,
        chunk_policy: TtsChunkPolicy::Adaptive,
        ..TtsRunConfig::default()
    };
    let mut controller = AdaptiveChunkController::new(&config);
    assert_eq!(controller.target_frames(&config), 16);

    controller.observe(16, std::time::Duration::from_millis(32), 16);
    let first = controller.target_frames(&config);
    assert!(first >= 16);

    let near_first = ((first as f64) * (1.0 + DEFAULT_CHUNK_HYSTERESIS_FRACTION / 2.0)).round() as usize;
    controller.current_chunk_frames = first;
    controller.ema_ms_per_frame = Some(config.target_emit_latency_ms as f64 / near_first as f64);
    assert_eq!(controller.target_frames(&config), first);

    let far_target =
        ((first as f64) * (1.0 + DEFAULT_CHUNK_HYSTERESIS_FRACTION + DEFAULT_CHUNK_EMA_ALPHA)).round() as usize;
    controller.ema_ms_per_frame = Some(config.target_emit_latency_ms as f64 / far_target as f64);
    assert_ne!(controller.target_frames(&config), first);
}

#[test]
fn adaptive_chunk_controller_does_not_shrink_mid_run() {
    let config = TtsRunConfig {
        min_chunk_frames: 16,
        max_chunk_frames: 256,
        target_emit_latency_ms: 80,
        chunk_policy: TtsChunkPolicy::Adaptive,
        ..TtsRunConfig::default()
    };
    let mut controller = AdaptiveChunkController::new(&config);
    controller.current_chunk_frames = 96;
    controller.ema_ms_per_frame = Some(40.0);
    assert_eq!(controller.target_frames(&config), 96);
}

#[test]
fn startup_target_frames_backoff_caps_at_startup_cap() {
    assert_eq!(super::next_startup_target_frames(1, 64), 2);
    assert_eq!(super::next_startup_target_frames(2, 64), 4);
    assert_eq!(super::next_startup_target_frames(8, 64), 16);
    assert_eq!(super::next_startup_target_frames(32, 64), 64);
    assert_eq!(super::next_startup_target_frames(64, 64), 64);
    assert_eq!(super::next_startup_target_frames(128, 64), 64);
}

#[test]
fn vocoder_streaming_mode_maps_to_audio_decode_mode() {
    let incremental = TtsRunConfig {
        vocoder_streaming_mode: TtsVocoderStreamingMode::IncrementalStateful,
        ..TtsRunConfig::default()
    };
    assert_eq!(audio_decode_streaming_mode(&incremental), AudioDecodeStreamingMode::IncrementalStateful);

    let prefix = TtsRunConfig {
        vocoder_streaming_mode: TtsVocoderStreamingMode::PrefixFallback,
        max_stream_workspace_frames: 1024,
        max_semantic_frames: 1024,
        ..TtsRunConfig::default()
    };
    assert_eq!(audio_decode_streaming_mode(&prefix), AudioDecodeStreamingMode::PrefixFallback);
}

#[test]
fn adaptive_chunk_controller_scales_up_when_decode_lags_realtime() {
    let config = TtsRunConfig {
        min_chunk_frames: 16,
        max_chunk_frames: 256,
        ..TtsRunConfig::default()
    };
    let mut controller = AdaptiveChunkController::new(&config);
    controller.current_chunk_frames = 16;

    controller.adapt_up_for_realtime(&config, 16, 44_100, std::time::Duration::from_millis(500), 8_192);
    assert!(controller.current_chunk_frames > 16);

    let after_up = controller.current_chunk_frames;
    controller.adapt_up_for_realtime(&config, 16, 44_100, std::time::Duration::from_millis(20), 8_192);
    assert_eq!(controller.current_chunk_frames, after_up);
}

#[test]
fn adaptive_chunk_controller_promotes_to_max_chunk() {
    let config = TtsRunConfig {
        min_chunk_frames: 16,
        max_chunk_frames: 256,
        ..TtsRunConfig::default()
    };
    let mut controller = AdaptiveChunkController::new(&config);
    controller.current_chunk_frames = 32;
    controller.promote_to_max_chunk(&config);
    assert_eq!(controller.current_chunk_frames, 256);
}

#[test]
fn pending_stream_chunk_timing_excludes_idle_overlap_between_submit_and_flush() {
    struct TimedPendingAudioChunk {
        pcm: Option<AudioPcmBatch>,
        resolve_duration: std::time::Duration,
        step_stats: Option<AudioDecodeStepStats>,
    }

    impl PendingAudioChunkBackend for TimedPendingAudioChunk {
        fn is_ready(&self) -> bool {
            true
        }

        fn step_stats(&self) -> Option<AudioDecodeStepStats> {
            self.step_stats
        }

        fn resolve(mut self: Box<Self>) -> Result<AudioPcmBatch, Error> {
            self.pcm.take().ok_or(Error::GenerateFailed)
        }

        fn resolve_with_decode_duration(
            mut self: Box<Self>
        ) -> Result<(AudioPcmBatch, std::time::Duration), Error> {
            let pcm = self.pcm.take().ok_or(Error::GenerateFailed)?;
            Ok((pcm, self.resolve_duration))
        }
    }

    let pcm =
        AudioPcmBatch::new(vec![0.0_f32; 2_400].into_boxed_slice(), 24_000, 1, vec![2_400].into_boxed_slice())
            .expect("pcm");
    let mut pending_chunk = Some(PendingStreamingChunk {
        submission_decode_duration: std::time::Duration::from_millis(11),
        ready_frames: 4,
        next_chunk_frames: 8,
        chunk: Box::new(TimedPendingAudioChunk {
            pcm: Some(pcm),
            resolve_duration: std::time::Duration::from_millis(7),
            step_stats: Some(AudioDecodeStepStats {
                input_frames: 4,
                total_semantic_frames: 4,
                decoded_window_start_frame: 0,
                decoded_window_frames: 4,
            }),
        }),
    });
    std::thread::sleep(std::time::Duration::from_millis(50));

    let config = TtsRunConfig {
        min_chunk_frames: 4,
        max_chunk_frames: 64,
        chunk_policy: TtsChunkPolicy::Adaptive,
        ..TtsRunConfig::default()
    };
    let mut on_chunk = |_pcm: &AudioPcmBatch| {};
    let mut callback_seconds = 0.0_f64;
    let mut audio_decode_seconds_in_loop = 0.0_f64;
    let mut audio_decode_seconds = 0.0_f64;
    let mut output_samples = Vec::new();
    let mut output_frames = 0usize;
    let mut output_sample_rate = 0u32;
    let mut output_channels = 0usize;
    let mut emitted_chunks = 0usize;
    let mut first_emit_pending = false;
    let mut first_chunk_seconds = None;
    let mut first_chunk_frames = 0usize;
    let stream_start = std::time::Instant::now();
    let mut startup_target_frames = 4usize;
    let startup_cap_frames = 64usize;
    let mut chunk_controller = AdaptiveChunkController::new(&config);
    let mut audio_decode_calls = 0usize;
    let mut audio_input_frames = 0usize;
    let mut audio_decoded_window_frames = 0usize;
    let mut audio_max_decoded_window_frames = 0usize;

    maybe_flush_pending_stream_chunk(
        &mut pending_chunk,
        false,
        true,
        &mut on_chunk,
        &mut callback_seconds,
        &mut audio_decode_seconds_in_loop,
        &mut audio_decode_seconds,
        &mut output_samples,
        &mut output_frames,
        &mut output_sample_rate,
        &mut output_channels,
        &mut emitted_chunks,
        &mut first_emit_pending,
        &mut first_chunk_seconds,
        &mut first_chunk_frames,
        stream_start,
        &mut startup_target_frames,
        startup_cap_frames,
        &mut chunk_controller,
        &config,
        &mut audio_decode_calls,
        &mut audio_input_frames,
        &mut audio_decoded_window_frames,
        &mut audio_max_decoded_window_frames,
    )
    .expect("flush pending chunk");

    assert!(pending_chunk.is_none());
    assert!((audio_decode_seconds - 0.018).abs() < 1e-9);
    assert!((audio_decode_seconds_in_loop - 0.018).abs() < 1e-9);
    assert_eq!(audio_decode_calls, 1);
    assert_eq!(audio_input_frames, 4);
    assert_eq!(audio_decoded_window_frames, 4);
    assert_eq!(audio_max_decoded_window_frames, 4);
    assert_eq!(emitted_chunks, 1);
    assert_eq!(output_frames, 2_400);
    assert_eq!(output_sample_rate, 24_000);
    assert_eq!(output_channels, 1);
    assert_eq!(output_samples.len(), 2_400);
    assert_eq!(chunk_controller.current_chunk_frames, 8);
    assert_eq!(chunk_controller.ema_ms_per_frame, Some(4.5));
}
