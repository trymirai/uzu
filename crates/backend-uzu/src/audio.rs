//! Host-side audio transforms shared by speech model backends.

use std::{f32::consts::PI, sync::Arc};

use half::f16;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use shoji::types::basic::PcmBatch;
use thiserror::Error;

const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const N_SAMPLES: usize = 30 * SAMPLE_RATE as usize;
const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH;

/// Invalid PCM or mel geometry for the fixed Whisper audio frontend.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum WhisperAudioError {
    #[error("Whisper requires 16 kHz PCM, got {0} Hz")]
    UnsupportedSampleRate(u32),
    #[error("PCM channel count must be greater than zero")]
    MissingChannels,
    #[error("PCM sample count is not divisible by its channel count")]
    IncompleteFrame,
    #[error("Whisper requires exactly one PCM batch item, got {0}")]
    ExpectedSingleItem(usize),
    #[error("PCM batch declares {declared} frames but contains {actual}")]
    LengthMismatch {
        declared: usize,
        actual: usize,
    },
    #[error("Whisper accepts at most one 30-second chunk per transform")]
    ChunkTooLong,
    #[error("PCM sample at index {0} must be finite")]
    NonFiniteSample(usize),
    #[error("PCM sample at index {0} must be normalized to [-1, 1]")]
    OutOfRangeSample(usize),
    #[error("Whisper checkpoints support 80 or 128 mel bins, got {0}")]
    UnsupportedMelBins(usize),
}

/// Whisper encoder input in frame-major half-precision layout.
#[derive(Debug, PartialEq)]
pub struct WhisperLogMelSpectrogram {
    values: Box<[f16]>,
    mel_bin_count: usize,
}

impl WhisperLogMelSpectrogram {
    /// Frames retained after Whisper omits the final centered STFT frame.
    pub const FRAME_COUNT: usize = N_FRAMES;

    /// Number of mel bins in each frame.
    pub fn mel_bin_count(&self) -> usize {
        self.mel_bin_count
    }

    /// Frame-major `[frame_count, mel_bin_count]` dimensions.
    pub fn shape(&self) -> [usize; 2] {
        [Self::FRAME_COUNT, self.mel_bin_count]
    }

    /// Frame-major half-precision values.
    pub fn as_slice(&self) -> &[f16] {
        &self.values
    }

    /// Frame-major half-precision values for transfer into a backend allocation.
    pub fn into_values(self) -> Box<[f16]> {
        self.values
    }
}

/// Reusable OpenAI Whisper log-mel transform and its fixed-size workspace.
///
/// Construction retains the checkpoint-specific Slaney filter bank, periodic
/// Hann window, RustFFT algorithm, and scratch allocations across calls. One
/// mutable frontend should therefore be owned by each independently executing
/// audio stream.
pub struct WhisperLogMelFrontend {
    mel_bin_count: usize,
    filters: Box<[f32]>,
    window: Box<[f32]>,
    fft: Arc<dyn Fft<f32>>,
    audio: Box<[f32]>,
    spectrum: Box<[Complex<f32>]>,
    magnitudes: Box<[f32]>,
    mel: Box<[f32]>,
}

impl WhisperLogMelFrontend {
    /// Frontend geometry for one 80- or 128-bin Whisper checkpoint.
    pub fn new(mel_bin_count: usize) -> Result<Self, WhisperAudioError> {
        if !matches!(mel_bin_count, 80 | 128) {
            return Err(WhisperAudioError::UnsupportedMelBins(mel_bin_count));
        }

        let filters = create_mel_filters(mel_bin_count);
        let window =
            (0..N_FFT).map(|index| 0.5 - 0.5 * (2.0 * PI * index as f32 / N_FFT as f32).cos()).collect::<Box<[_]>>();
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let frequency_bin_count = N_FFT / 2 + 1;

        Ok(Self {
            mel_bin_count,
            filters,
            window,
            fft,
            audio: vec![0.0; N_SAMPLES].into_boxed_slice(),
            spectrum: vec![Complex::new(0.0, 0.0); N_FFT].into_boxed_slice(),
            magnitudes: vec![0.0; frequency_bin_count].into_boxed_slice(),
            mel: vec![0.0; N_FRAMES * mel_bin_count].into_boxed_slice(),
        })
    }

    /// Number of mel bins selected by the checkpoint geometry.
    pub fn mel_bin_count(&self) -> usize {
        self.mel_bin_count
    }

    /// Produces Whisper's fixed-shape log-mel encoder input.
    ///
    /// The input must contain exactly one normalized 16 kHz PCM item of at most
    /// 30 seconds. Interleaved channels are combined with an unweighted
    /// arithmetic mean because [`PcmBatch`] carries no channel-layout metadata;
    /// callers with nonuniform channel layouts should downmix first. Short
    /// inputs are zero-padded to 480,000 samples.
    ///
    /// The transform applies Whisper's centered periodic-Hann STFT, Slaney mel
    /// filters, and dynamic-range normalization. Like OpenAI's frontend, it
    /// omits the final centered STFT frame and returns `f16` values in
    /// frame-major `[3_000, mel_bin_count]` layout.
    pub fn transform(
        &mut self,
        pcm: &PcmBatch,
    ) -> Result<WhisperLogMelSpectrogram, WhisperAudioError> {
        validate_pcm_batch(pcm)?;
        self.downmix_and_pad(pcm);

        let frequency_bin_count = self.magnitudes.len();
        for frame_index in 0..N_FRAMES {
            let start = frame_index as isize * HOP_LENGTH as isize - (N_FFT / 2) as isize;
            for (window_index, value) in self.spectrum.iter_mut().enumerate() {
                let audio_index = reflect_index(start + window_index as isize, self.audio.len());
                *value = Complex::new(self.audio[audio_index] * self.window[window_index], 0.0);
            }

            self.fft.process(&mut self.spectrum);
            for (magnitude, value) in self.magnitudes.iter_mut().zip(&self.spectrum) {
                *magnitude = value.norm_sqr();
            }

            for mel_index in 0..self.mel_bin_count {
                let filter = &self.filters[mel_index * frequency_bin_count..(mel_index + 1) * frequency_bin_count];
                let energy = filter.iter().zip(&self.magnitudes).map(|(weight, value)| weight * value).sum::<f32>();
                self.mel[frame_index * self.mel_bin_count + mel_index] = energy.max(1.0e-10).log10();
            }
        }

        let maximum = self.mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let floor = maximum - 8.0;
        let values = self.mel.iter().map(|value| f16::from_f32((value.max(floor) + 4.0) / 4.0)).collect();
        Ok(WhisperLogMelSpectrogram {
            values,
            mel_bin_count: self.mel_bin_count,
        })
    }

    /// Equal-channel downmix followed by Whisper's fixed 30-second zero padding.
    fn downmix_and_pad(
        &mut self,
        pcm: &PcmBatch,
    ) {
        self.audio.fill(0.0);
        for (frame_index, frame) in pcm.samples.chunks_exact(pcm.channels as usize).enumerate() {
            let sum = frame.iter().sum::<f64>();
            self.audio[frame_index] = (sum / pcm.channels as f64) as f32;
        }
    }
}

/// Single-chunk shape and value contract required by Whisper's frontend.
fn validate_pcm_batch(pcm: &PcmBatch) -> Result<(), WhisperAudioError> {
    if pcm.sample_rate != SAMPLE_RATE {
        return Err(WhisperAudioError::UnsupportedSampleRate(pcm.sample_rate));
    }
    if pcm.channels == 0 {
        return Err(WhisperAudioError::MissingChannels);
    }
    if !pcm.samples.len().is_multiple_of(pcm.channels as usize) {
        return Err(WhisperAudioError::IncompleteFrame);
    }
    if pcm.lengths.len() != 1 {
        return Err(WhisperAudioError::ExpectedSingleItem(pcm.lengths.len()));
    }

    let frame_count = pcm.samples.len() / pcm.channels as usize;
    let declared_frame_count = pcm.lengths[0] as usize;
    if declared_frame_count != frame_count {
        return Err(WhisperAudioError::LengthMismatch {
            declared: declared_frame_count,
            actual: frame_count,
        });
    }
    if frame_count > N_SAMPLES {
        return Err(WhisperAudioError::ChunkTooLong);
    }
    for (sample_index, sample) in pcm.samples.iter().enumerate() {
        if !sample.is_finite() {
            return Err(WhisperAudioError::NonFiniteSample(sample_index));
        }
        if sample.abs() > 1.0 {
            return Err(WhisperAudioError::OutOfRangeSample(sample_index));
        }
    }
    Ok(())
}

fn reflect_index(
    mut index: isize,
    length: usize,
) -> usize {
    let length = length as isize;
    while index < 0 || index >= length {
        index = if index < 0 {
            -index
        } else {
            2 * length - 2 - index
        };
    }
    index as usize
}

/// Librosa's default Slaney bank, which OpenAI serialized as `mel_filters.npz`.
fn create_mel_filters(mel_bin_count: usize) -> Box<[f32]> {
    let frequency_bin_count = N_FFT / 2 + 1;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(SAMPLE_RATE as f32 / 2.0);
    let mel_frequencies = (0..mel_bin_count + 2)
        .map(|index| {
            let fraction = index as f32 / (mel_bin_count + 1) as f32;
            mel_to_hz(mel_min + fraction * (mel_max - mel_min))
        })
        .collect::<Box<[_]>>();

    let mut filters = vec![0.0f32; mel_bin_count * frequency_bin_count];
    for mel_index in 0..mel_bin_count {
        let lower = mel_frequencies[mel_index];
        let center = mel_frequencies[mel_index + 1];
        let upper = mel_frequencies[mel_index + 2];
        let normalization = 2.0 / (upper - lower);
        for frequency_index in 0..frequency_bin_count {
            let frequency = frequency_index as f32 * SAMPLE_RATE as f32 / N_FFT as f32;
            let lower_slope = (frequency - lower) / (center - lower);
            let upper_slope = (upper - frequency) / (upper - center);
            filters[mel_index * frequency_bin_count + frequency_index] =
                lower_slope.min(upper_slope).max(0.0) * normalization;
        }
    }
    filters.into_boxed_slice()
}

fn hz_to_mel(frequency: f32) -> f32 {
    const LINEAR_SCALE: f32 = 200.0 / 3.0;
    const LOG_FREQUENCY: f32 = 1_000.0;
    const LOG_MEL: f32 = LOG_FREQUENCY / LINEAR_SCALE;
    const LOG_STEP: f32 = 0.068_751_78;

    if frequency < LOG_FREQUENCY {
        frequency / LINEAR_SCALE
    } else {
        LOG_MEL + (frequency / LOG_FREQUENCY).ln() / LOG_STEP
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    const LINEAR_SCALE: f32 = 200.0 / 3.0;
    const LOG_FREQUENCY: f32 = 1_000.0;
    const LOG_MEL: f32 = LOG_FREQUENCY / LINEAR_SCALE;
    const LOG_STEP: f32 = 0.068_751_78;

    if mel < LOG_MEL {
        mel * LINEAR_SCALE
    } else {
        LOG_FREQUENCY * (LOG_STEP * (mel - LOG_MEL)).exp()
    }
}

#[cfg(test)]
#[path = "../tests/unit/audio_test.rs"]
mod tests;
