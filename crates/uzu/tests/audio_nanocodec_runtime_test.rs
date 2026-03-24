#![cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]

mod common;

use common::audio_nanocodec_fsq_reference::{fsq_decode_reference, fsq_encode_reference};
use uzu::{
    NanoCodecFsqRuntime, NanoCodecFsqRuntimeConfig,
    backends::metal::Metal,
    prelude::{AudioCodecRuntime, AudioError, AudioPcmBatch, AudioTokenGrid},
};

fn create_runtime() -> NanoCodecFsqRuntime<Metal> {
    let config =
        NanoCodecFsqRuntimeConfig::new(24_000, 2, vec![8, 6].into_boxed_slice(), 1e-3).expect("valid runtime config");
    NanoCodecFsqRuntime::new(config)
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
    let runtime = create_runtime();
    let pcm = make_pcm(runtime.config().channels(), &[3, 5]);
    let tokens = runtime.encode(&pcm).expect("encode");

    let (padded_input, lengths_i32, frames) = pack_for_reference(&pcm);
    let expected_i32 = fsq_encode_reference(
        &padded_input,
        &lengths_i32,
        pcm.batch_size(),
        runtime.config().num_groups(),
        frames,
        runtime.config().codebook_dim_per_group(),
        runtime.config().num_levels_per_group(),
        runtime.config().dim_base_index(),
        runtime.config().eps(),
    )
    .expect("fsq reference encode");
    let expected_u32: Vec<u32> = expected_i32.into_iter().map(|value| value as u32).collect();
    let expected = AudioTokenGrid::new(
        expected_u32.into_boxed_slice(),
        pcm.batch_size(),
        runtime.config().num_groups(),
        frames,
        pcm.lengths().to_vec().into_boxed_slice(),
    )
    .expect("expected token grid");

    assert_eq!(tokens.batch_size(), expected.batch_size());
    assert_eq!(tokens.codebooks(), expected.codebooks());
    assert_eq!(tokens.frames(), expected.frames());
    assert_eq!(tokens.lengths(), expected.lengths());
    assert_eq!(tokens.tokens(), expected.tokens());
}

#[test]
fn nanocodec_runtime_decode_matches_fsq_reference() {
    let runtime = create_runtime();
    let pcm = make_pcm(runtime.config().channels(), &[4, 2]);
    let encoded = runtime.encode(&pcm).expect("encode");
    let decoded = runtime.decode(&encoded).expect("decode");

    let reference_tokens_i32: Vec<i32> = encoded.tokens().iter().map(|&token| token as i32).collect();
    let lengths_i32: Vec<i32> = encoded.lengths().iter().map(|&length| length as i32).collect();
    let reference_padded = fsq_decode_reference(
        &reference_tokens_i32,
        &lengths_i32,
        encoded.batch_size(),
        runtime.config().num_groups(),
        encoded.frames(),
        runtime.config().codebook_dim_per_group(),
        runtime.config().num_levels_per_group(),
    )
    .expect("fsq reference decode");
    let expected_samples =
        unpack_reference_output(&reference_padded, runtime.config().channels(), encoded.frames(), encoded.lengths());

    assert_eq!(decoded.sample_rate(), runtime.config().sample_rate());
    assert_eq!(decoded.channels(), runtime.config().channels());
    assert_eq!(decoded.lengths(), encoded.lengths());

    for (index, (&got, &expected)) in decoded.samples().iter().zip(expected_samples.iter()).enumerate() {
        let delta = (got - expected).abs();
        assert!(delta <= 1e-5_f32, "decode mismatch at index={index}: got={got}, expected={expected}, delta={delta}");
    }
}

#[test]
fn nanocodec_runtime_decode_rejects_out_of_range_token() {
    let runtime = create_runtime();
    let cardinality = runtime.config().codec_cardinality();
    let tokens = AudioTokenGrid::new(
        vec![0, cardinality, 1, 2].into_boxed_slice(),
        1,
        runtime.config().num_groups(),
        2,
        vec![2usize].into_boxed_slice(),
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
    let runtime = create_runtime();
    let pcm = make_pcm(runtime.config().channels() - 1, &[2]);

    let error = runtime.encode(&pcm).expect_err("encode should fail");
    match error {
        AudioError::Runtime(message) => {
            assert!(message.contains("channel mismatch"), "unexpected error message: {message}");
        },
        other => panic!("unexpected error variant: {other:?}"),
    }
}
