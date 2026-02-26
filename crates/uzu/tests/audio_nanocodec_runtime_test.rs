#![cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]

use uzu::audio::{
    AudioCodecRuntime, AudioError, AudioPcmBatch, AudioTokenGrid, AudioTokenPacking, NanoCodecFsqRuntime,
    NanoCodecFsqRuntimeConfig,
    nanocodec::fsq::{compute_dim_base_index, fsq_decode_reference, fsq_encode_reference},
};

fn create_runtime(output_packing: AudioTokenPacking) -> NanoCodecFsqRuntime {
    let config = NanoCodecFsqRuntimeConfig::new(24_000, 2, vec![8, 6].into_boxed_slice(), 1e-3, output_packing)
        .expect("valid runtime config");
    NanoCodecFsqRuntime::new(config)
}

fn create_runtime_with_decoder() -> NanoCodecFsqRuntime {
    let tts_config = serde_json::json!({
        "audio_codec": {
            "type": "nanocodec_fsq",
            "sample_rate": 24000,
            "num_groups": 2,
            "num_levels_per_group": [8, 6],
            "output_packing": "codebook_major",
            "decoder": {
                "pre_conv": {
                    "weight": {
                        "shape": [2, 4, 1],
                        "values": [
                            1.0, 0.0, 0.0, 0.0,
                            0.0, 1.0, 0.0, 0.0
                        ]
                    },
                    "bias": [0.0, 0.0]
                },
                "stages": [{
                    "activation_alpha": [1.0],
                    "upsample_conv": {
                        "weight": {
                            "shape": [2, 1, 4],
                            "values": [
                                1.0, 0.0, 0.0, 0.0,
                                1.0, 0.0, 0.0, 0.0
                            ]
                        },
                        "bias": [0.0, 0.0],
                        "stride": 2,
                        "groups": 2
                    }
                }],
                "post_activation_alpha": [1.0],
                "post_conv": {
                    "weight": {
                        "shape": [1, 2, 1],
                        "values": [1.0, 0.0]
                    },
                    "bias": [0.0]
                }
            }
        }
    });

    NanoCodecFsqRuntime::from_tts_config_value(&tts_config).expect("runtime with decoder")
}

fn create_runtime_with_residual_decoder() -> NanoCodecFsqRuntime {
    let tts_config = serde_json::json!({
        "audio_codec": {
            "type": "nanocodec_fsq",
            "sample_rate": 24000,
            "num_groups": 2,
            "num_levels_per_group": [8, 6],
            "output_packing": "codebook_major",
            "decoder": {
                "pre_conv": {
                    "weight": {
                        "shape": [2, 4, 1],
                        "values": [
                            1.0, 0.0, 0.0, 0.0,
                            0.0, 1.0, 0.0, 0.0
                        ]
                    },
                    "bias": [0.0, 0.0]
                },
                "stages": [{
                    "activation_alpha": [1.0],
                    "upsample_conv": {
                        "weight": {
                            "shape": [2, 1, 4],
                            "values": [
                                1.0, 0.0, 0.0, 0.0,
                                1.0, 0.0, 0.0, 0.0
                            ]
                        },
                        "bias": [0.0, 0.0],
                        "stride": 2,
                        "groups": 2
                    },
                    "res_layer": {
                        "res_blocks": [{
                            "res_blocks": [{
                                "input_activation_alpha": [1.0],
                                "input_conv": {
                                    "weight": {
                                        "shape": [2, 2, 1],
                                        "values": [0.0, 0.0, 0.0, 0.0]
                                    },
                                    "bias": [0.0, 0.0]
                                },
                                "skip_activation_alpha": [1.0],
                                "skip_conv": {
                                    "weight": {
                                        "shape": [2, 2, 1],
                                        "values": [0.0, 0.0, 0.0, 0.0]
                                    },
                                    "bias": [0.5, 0.0]
                                }
                            }]
                        }]
                    }
                }],
                "post_activation_alpha": [1.0],
                "post_conv": {
                    "weight": {
                        "shape": [1, 2, 1],
                        "values": [1.0, 0.0]
                    },
                    "bias": [0.0]
                }
            }
        }
    });

    NanoCodecFsqRuntime::from_tts_config_value(&tts_config).expect("runtime with residual decoder")
}

fn make_pcm(
    channels: usize,
    lengths: &[usize],
) -> AudioPcmBatch {
    let sample_count = lengths.iter().sum::<usize>() * channels;
    let samples: Vec<f32> = (0..sample_count).map(|index| (index as f32 * 0.071 - 1.3).sin() * 0.9).collect();
    AudioPcmBatch::new(samples.into_boxed_slice(), 24_000, channels, lengths.to_vec().into_boxed_slice()).expect("pcm")
}

fn pack_for_reference(pcm: &AudioPcmBatch) -> (Vec<f32>, Vec<i32>, usize) {
    let batch_size = pcm.batch_size();
    let channels = pcm.channels();
    let frames = pcm.lengths().iter().copied().max().unwrap_or(0);
    let mut padded = vec![0.0_f32; batch_size * channels * frames];
    let mut lengths_i32 = Vec::with_capacity(batch_size);

    let mut src_offset = 0usize;
    for (batch, &length) in pcm.lengths().iter().enumerate() {
        lengths_i32.push(length as i32);
        for frame in 0..length {
            for channel in 0..channels {
                let src_index = src_offset + frame * channels + channel;
                let dst_index = (batch * channels + channel) * frames + frame;
                padded[dst_index] = pcm.samples()[src_index];
            }
        }
        src_offset += length * channels;
    }

    (padded, lengths_i32, frames)
}

fn unpack_reference_output(
    padded: &[f32],
    channels: usize,
    frames: usize,
    lengths: &[usize],
) -> Vec<f32> {
    let total = lengths.iter().sum::<usize>() * channels;
    let mut out = vec![0.0_f32; total];
    let mut dst_offset = 0usize;

    for (batch, &length) in lengths.iter().enumerate() {
        for frame in 0..length {
            for channel in 0..channels {
                let src_index = (batch * channels + channel) * frames + frame;
                let dst_index = dst_offset + frame * channels + channel;
                out[dst_index] = padded[src_index];
            }
        }
        dst_offset += length * channels;
    }

    out
}

#[test]
fn nanocodec_runtime_encode_matches_fsq_reference() {
    let runtime = create_runtime(AudioTokenPacking::FrameMajor);
    let pcm = make_pcm(runtime.config().channels(), &[3, 5]);
    let tokens = runtime.encode(&pcm).expect("encode");

    let (padded_input, lengths_i32, frames) = pack_for_reference(&pcm);
    let dim_base_index = compute_dim_base_index(runtime.config().num_levels_per_group()).expect("dim base index");
    let expected_i32 = fsq_encode_reference(
        &padded_input,
        &lengths_i32,
        pcm.batch_size(),
        runtime.config().num_groups(),
        frames,
        runtime.config().codebook_dim_per_group(),
        runtime.config().num_levels_per_group(),
        &dim_base_index,
        runtime.config().eps(),
    )
    .expect("fsq reference encode");
    let expected_u32: Vec<u32> = expected_i32.into_iter().map(|value| value as u32).collect();
    let expected_codebook_major = AudioTokenGrid::new(
        expected_u32.into_boxed_slice(),
        pcm.batch_size(),
        runtime.config().num_groups(),
        frames,
        pcm.lengths().to_vec().into_boxed_slice(),
        AudioTokenPacking::CodebookMajor,
    )
    .expect("expected token grid");
    let expected = expected_codebook_major.to_packing(AudioTokenPacking::FrameMajor);

    assert_eq!(tokens.batch_size(), expected.batch_size());
    assert_eq!(tokens.codebooks(), expected.codebooks());
    assert_eq!(tokens.frames(), expected.frames());
    assert_eq!(tokens.lengths(), expected.lengths());
    assert_eq!(tokens.packing(), AudioTokenPacking::FrameMajor);
    assert_eq!(tokens.tokens(), expected.tokens());
}

#[test]
fn nanocodec_runtime_decode_matches_fsq_reference() {
    let runtime = create_runtime(AudioTokenPacking::FrameMajor);
    let pcm = make_pcm(runtime.config().channels(), &[4, 2]);
    let encoded = runtime.encode(&pcm).expect("encode");
    let decoded = runtime.decode(&encoded).expect("decode");

    let encoded_codebook_major = encoded.to_packing(AudioTokenPacking::CodebookMajor);
    let reference_tokens_i32: Vec<i32> = encoded_codebook_major.tokens().iter().map(|&token| token as i32).collect();
    let lengths_i32: Vec<i32> = encoded_codebook_major.lengths().iter().map(|&length| length as i32).collect();
    let reference_padded = fsq_decode_reference(
        &reference_tokens_i32,
        &lengths_i32,
        encoded_codebook_major.batch_size(),
        runtime.config().num_groups(),
        encoded_codebook_major.frames(),
        runtime.config().codebook_dim_per_group(),
        runtime.config().num_levels_per_group(),
    )
    .expect("fsq reference decode");
    let expected_samples = unpack_reference_output(
        &reference_padded,
        runtime.config().channels(),
        encoded_codebook_major.frames(),
        encoded_codebook_major.lengths(),
    );

    assert_eq!(decoded.sample_rate(), runtime.config().sample_rate());
    assert_eq!(decoded.channels(), runtime.config().channels());
    assert_eq!(decoded.lengths(), encoded.lengths());

    for (index, (&got, &expected)) in decoded.samples().iter().zip(expected_samples.iter()).enumerate() {
        let delta = (got - expected).abs();
        assert!(delta <= 1e-5, "decode mismatch at index={index}: got={got}, expected={expected}, delta={delta}");
    }
}

#[test]
fn nanocodec_runtime_decode_rejects_out_of_range_token() {
    let runtime = create_runtime(AudioTokenPacking::CodebookMajor);
    let cardinality = runtime.config().codec_cardinality();
    let tokens = AudioTokenGrid::new(
        vec![0, cardinality, 1, 2].into_boxed_slice(),
        1,
        runtime.config().num_groups(),
        2,
        vec![2usize].into_boxed_slice(),
        AudioTokenPacking::CodebookMajor,
    )
    .expect("token grid");

    let error = runtime.decode(&tokens).expect_err("decode should fail");
    assert_eq!(
        error,
        AudioError::InvalidCodecToken {
            token: cardinality,
            cardinality,
        }
    );
}

#[test]
fn nanocodec_runtime_encode_rejects_channel_mismatch() {
    let runtime = create_runtime(AudioTokenPacking::CodebookMajor);
    let pcm = make_pcm(runtime.config().channels() - 1, &[2]);

    let error = runtime.encode(&pcm).expect_err("encode should fail");
    match error {
        AudioError::Runtime(message) => {
            assert!(message.contains("channel mismatch"), "unexpected error message: {message}");
        },
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn nanocodec_runtime_with_decoder_scales_decode_lengths_and_channels() {
    let runtime = create_runtime_with_decoder();
    let tokens = AudioTokenGrid::new(
        vec![0, 7, 11, 3, 5, 9].into_boxed_slice(),
        1,
        runtime.config().num_groups(),
        3,
        vec![2usize].into_boxed_slice(),
        AudioTokenPacking::CodebookMajor,
    )
    .expect("token grid");

    let decoded = runtime.decode(&tokens).expect("decode");
    assert_eq!(decoded.channels(), 1);
    assert_eq!(decoded.lengths(), &[4usize]);
    assert_eq!(decoded.sample_rate(), 24_000);
}

#[test]
fn nanocodec_runtime_with_decoder_rejects_encode() {
    let runtime = create_runtime_with_decoder();
    let pcm = make_pcm(1, &[2]);

    let error = runtime.encode(&pcm).expect_err("encode should fail");
    match error {
        AudioError::Runtime(message) => {
            assert!(message.contains("not supported"), "unexpected message: {message}");
        },
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn nanocodec_runtime_with_residual_decoder_changes_waveform() {
    let baseline_runtime = create_runtime_with_decoder();
    let residual_runtime = create_runtime_with_residual_decoder();
    let tokens = AudioTokenGrid::new(
        vec![0, 7, 11, 3, 5, 9].into_boxed_slice(),
        1,
        baseline_runtime.config().num_groups(),
        3,
        vec![2usize].into_boxed_slice(),
        AudioTokenPacking::CodebookMajor,
    )
    .expect("token grid");

    let baseline = baseline_runtime.decode(&tokens).expect("baseline decode");
    let residual = residual_runtime.decode(&tokens).expect("residual decode");

    assert_eq!(baseline.sample_rate(), residual.sample_rate());
    assert_eq!(baseline.channels(), residual.channels());
    assert_eq!(baseline.lengths(), residual.lengths());

    let max_delta = baseline
        .samples()
        .iter()
        .zip(residual.samples().iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_delta > 1e-5, "residual decoder should alter waveform");
}
