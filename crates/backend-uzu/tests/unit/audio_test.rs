use proc_macros::uzu_test;
use shoji::types::basic::PcmBatch;

use super::{N_SAMPLES, WhisperAudioError, WhisperLogMelSpectrogram, validate_pcm_batch, whisper_log_mel_spectrogram};

const PCM_REFERENCE: &str = include_str!("../data/whisper/jfk_1s_200ms.s16le.hex");
const MEL_80_REFERENCE: &str = include_str!("../data/whisper/jfk_1s_200ms.mel80.tsv");
const MEL_128_REFERENCE: &str = include_str!("../data/whisper/jfk_1s_200ms.mel128.tsv");
const REFERENCE_FRAMES: [usize; 7] = [0, 4, 8, 12, 16, 20, 25];

fn pcm_batch(
    frame_count: usize,
    channels: u32,
    lengths: Vec<u32>,
) -> PcmBatch {
    PcmBatch {
        samples: vec![0.0; frame_count * channels as usize],
        sample_rate: 16_000,
        channels,
        lengths,
    }
}

fn pcm_error(pcm: &PcmBatch) -> WhisperAudioError {
    let result = whisper_log_mel_spectrogram(pcm, 128);
    result.expect_err("invalid PCM must be rejected before the transform")
}

#[uzu_test]
fn requires_whisper_sample_rate() {
    let mut pcm = pcm_batch(1, 1, vec![1]);
    pcm.sample_rate = 48_000;

    assert_eq!(pcm_error(&pcm), WhisperAudioError::UnsupportedSampleRate(48_000));
}

#[uzu_test]
fn rejects_missing_and_incomplete_channels() {
    let pcm = pcm_batch(0, 0, vec![0]);
    assert_eq!(pcm_error(&pcm), WhisperAudioError::MissingChannels);

    let pcm = PcmBatch {
        samples: vec![0.0; 3],
        sample_rate: 16_000,
        channels: 2,
        lengths: vec![1],
    };
    assert_eq!(pcm_error(&pcm), WhisperAudioError::IncompleteFrame);
}

#[uzu_test]
fn requires_exactly_one_batch_item() {
    let pcm = pcm_batch(0, 1, vec![]);
    assert_eq!(pcm_error(&pcm), WhisperAudioError::ExpectedSingleItem(0));

    let pcm = pcm_batch(2, 1, vec![1, 1]);
    assert_eq!(pcm_error(&pcm), WhisperAudioError::ExpectedSingleItem(2));
}

#[uzu_test]
fn rejects_non_finite_samples() {
    for sample in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut pcm = pcm_batch(2, 1, vec![2]);
        pcm.samples[1] = sample;

        assert_eq!(pcm_error(&pcm), WhisperAudioError::NonFiniteSample(1));
    }
}

#[uzu_test]
fn requires_normalized_samples() {
    let pcm = PcmBatch {
        samples: vec![-1.0, 1.0],
        sample_rate: 16_000,
        channels: 1,
        lengths: vec![2],
    };
    validate_pcm_batch(&pcm).expect("inclusive normalized boundaries");

    for sample in [1.0 + f64::EPSILON, -1.0 - f64::EPSILON, f64::MAX, f64::MIN] {
        let mut pcm = pcm_batch(2, 1, vec![2]);
        pcm.samples[1] = sample;

        assert_eq!(pcm_error(&pcm), WhisperAudioError::OutOfRangeSample(1));
    }
}

#[uzu_test]
fn rejects_unsupported_mel_bins() {
    let pcm = pcm_batch(0, 1, vec![0]);
    let result = whisper_log_mel_spectrogram(&pcm, 96);
    let error = result.expect_err("unsupported mel shape");

    assert_eq!(error, WhisperAudioError::UnsupportedMelBins(96));
}

#[uzu_test]
fn length_counts_interleaved_frames() {
    let pcm = pcm_batch(2, 2, vec![2]);
    validate_pcm_batch(&pcm).expect("two stereo frames");

    let pcm = pcm_batch(2, 2, vec![4]);
    assert_eq!(
        pcm_error(&pcm),
        WhisperAudioError::LengthMismatch {
            declared: 4,
            actual: 2,
        }
    );
}

#[uzu_test]
fn stereo_uses_an_unweighted_arithmetic_downmix() {
    let stereo = PcmBatch {
        samples: vec![-1.0, 0.0, 0.25, 0.75],
        sample_rate: 16_000,
        channels: 2,
        lengths: vec![2],
    };
    let mono = PcmBatch {
        samples: vec![-0.5, 0.5],
        sample_rate: 16_000,
        channels: 1,
        lengths: vec![2],
    };

    let stereo = whisper_log_mel_spectrogram(&stereo, 80).expect("valid stereo PCM");
    let mono = whisper_log_mel_spectrogram(&mono, 80).expect("valid mono PCM");
    assert_eq!(stereo, mono);
}

#[uzu_test]
fn enforces_the_inclusive_thirty_second_boundary() {
    let pcm = pcm_batch(0, 1, vec![0]);
    validate_pcm_batch(&pcm).expect("an empty single item is a valid padded chunk");

    let pcm = pcm_batch(N_SAMPLES, 1, vec![N_SAMPLES as u32]);
    validate_pcm_batch(&pcm).expect("exactly thirty seconds");

    let frame_count = N_SAMPLES + 1;
    let pcm = pcm_batch(frame_count, 1, vec![frame_count as u32]);
    assert_eq!(pcm_error(&pcm), WhisperAudioError::ChunkTooLong);
}

#[uzu_test]
fn whisper_log_mel_80_matches_openai() {
    assert_log_mel_matches(80, MEL_80_REFERENCE);
}

#[uzu_test]
fn whisper_log_mel_128_matches_openai() {
    assert_log_mel_matches(128, MEL_128_REFERENCE);
}

fn assert_log_mel_matches(
    mel_bin_count: usize,
    reference: &str,
) {
    let samples = decode_pcm_reference();
    assert_eq!(samples.len(), 3_200, "the captured 200 ms fixture must stay intact");
    let pcm = PcmBatch {
        lengths: vec![samples.len() as u32],
        samples,
        sample_rate: 16_000,
        channels: 1,
    };

    let actual = whisper_log_mel_spectrogram(&pcm, mel_bin_count).expect("valid reference audio");
    assert_eq!(actual.shape(), [WhisperLogMelSpectrogram::FRAME_COUNT, mel_bin_count]);
    assert_eq!(actual.mel_bin_count(), mel_bin_count);

    let mut reference_frames = Vec::new();
    let mut maximum_error = 0.0f32;
    let mut worst_coordinate = (0, 0);
    for line in reference.lines().filter(|line| !line.is_empty() && !line.starts_with('#')) {
        let (frame, expected) = line.split_once('\t').expect("frame and mel values");
        let frame = frame.parse::<usize>().expect("numeric frame index");
        reference_frames.push(frame);
        let expected = expected.split(',').map(|value| value.parse::<f32>().expect("numeric mel value"));

        let mut bin_count = 0;
        for (mel_bin, expected) in expected.enumerate() {
            let actual = actual.as_slice()[frame * mel_bin_count + mel_bin].to_f32();
            assert!(actual.is_finite(), "non-finite actual value at frame {frame}, mel bin {mel_bin}");
            assert!(expected.is_finite(), "non-finite reference at frame {frame}, mel bin {mel_bin}");
            let error = (actual - expected).abs();
            if error > maximum_error {
                maximum_error = error;
                worst_coordinate = (frame, mel_bin);
            }
            bin_count += 1;
        }
        assert_eq!(bin_count, mel_bin_count, "reference frame {frame}");
    }
    assert_eq!(reference_frames, REFERENCE_FRAMES);

    // Account for the production f16 boundary plus minor FFT implementation drift.
    assert!(
        maximum_error <= 0.002,
        "maximum error {maximum_error} at frame {}, mel bin {}",
        worst_coordinate.0,
        worst_coordinate.1
    );
    assert_eq!(actual.into_values().len(), WhisperLogMelSpectrogram::FRAME_COUNT * mel_bin_count);
}

fn decode_pcm_reference() -> Vec<f64> {
    let hex = PCM_REFERENCE.split_whitespace().collect::<String>();
    let (hex_bytes, remainder) = hex.as_bytes().as_chunks::<2>();
    assert!(remainder.is_empty(), "complete hexadecimal bytes");
    let bytes = hex_bytes
        .iter()
        .map(|digits| {
            let digits = std::str::from_utf8(digits).expect("ASCII hexadecimal");
            u8::from_str_radix(digits, 16).expect("hexadecimal byte")
        })
        .collect::<Vec<_>>();
    let (samples, remainder) = bytes.as_chunks::<2>();
    assert!(remainder.is_empty(), "complete signed 16-bit samples");

    samples.iter().map(|bytes| f64::from(i16::from_le_bytes(*bytes)) / 32_768.0).collect()
}
