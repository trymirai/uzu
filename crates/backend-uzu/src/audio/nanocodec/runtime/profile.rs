use std::sync::mpsc::Receiver;

use super::*;
use crate::{backends::common::Allocation, try_allocation_to_vec};

pub enum SubmittedDecodedPaddedAudio<B: Backend> {
    Ready(DecodedPaddedAudio),
    Pending {
        output: Allocation<B>,
        data_type: DataType,
        channels: usize,
        frames: usize,
        lengths: Vec<usize>,
        final_command_buffer: Pending<B>,
        completion_notification: Receiver<()>,
    },
}

impl<B: Backend> SubmittedDecodedPaddedAudio<B> {
    pub(crate) fn is_complete(&self) -> bool {
        use std::sync::mpsc::TryRecvError;

        match self {
            Self::Ready(_) => true,
            Self::Pending {
                completion_notification,
                ..
            } => match completion_notification.try_recv() {
                Ok(()) | Err(TryRecvError::Disconnected) => true,
                Err(TryRecvError::Empty) => false,
            },
        }
    }

    pub(in crate::audio::nanocodec::runtime) fn resolve(self) -> AudioResult<DecodedPaddedAudio> {
        match self {
            Self::Ready(decoded) => Ok(decoded),
            Self::Pending {
                output,
                data_type,
                channels,
                frames,
                lengths,
                final_command_buffer,
                completion_notification: _,
            } => {
                let completed_command_buffer = final_command_buffer.wait_until_completed().map_err(|err| {
                    AudioError::Runtime(format!("failed to wait for FishAudio decoder command buffer: {err}"))
                })?;
                let allocation_read_error =
                    |err| AudioError::Runtime(format!("failed to read FishAudio decoder output allocation: {err}"));
                let samples_result: AudioResult<Vec<f32>> = match data_type {
                    DataType::F32 => try_allocation_to_vec::<B, f32>(&output).map_err(allocation_read_error),
                    DataType::F16 => try_allocation_to_vec::<B, half::f16>(&output)
                        .map_err(allocation_read_error)
                        .map(|values| values.iter().map(|&value| f32::from(value)).collect()),
                    DataType::BF16 => try_allocation_to_vec::<B, half::bf16>(&output)
                        .map_err(allocation_read_error)
                        .map(|values| values.iter().map(|&value| f32::from(value)).collect()),
                    dt => Err(AudioError::Runtime(format!("unsupported vocoder output dtype: {dt:?}"))),
                };
                drop(output);
                drop(completed_command_buffer);
                let samples = samples_result?;
                Ok(DecodedPaddedAudio {
                    samples,
                    channels,
                    frames,
                    lengths,
                })
            },
        }
    }
}

pub(crate) struct PendingStreamPcmChunk<B: Backend> {
    pub(in crate::audio::nanocodec::runtime) runtime: NanoCodecFsqRuntime<B>,
    pub(in crate::audio::nanocodec::runtime) submitted: SubmittedDecodedPaddedAudio<B>,
    pub(in crate::audio::nanocodec::runtime) previous_audio_lengths: Box<[usize]>,
    pub(in crate::audio::nanocodec::runtime) semantic_lengths: Box<[usize]>,
    pub(in crate::audio::nanocodec::runtime) audio_offset_frames: usize,
    pub(in crate::audio::nanocodec::runtime) upsample_factor: usize,
    pub(in crate::audio::nanocodec::runtime) step_stats: AudioDecodeStepStats,
}

impl<B: Backend> PendingStreamPcmChunk<B> {
    pub(crate) fn step_stats(&self) -> AudioDecodeStepStats {
        self.step_stats
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.submitted.is_complete()
    }

    pub(crate) fn resolve(self) -> AudioResult<AudioPcmBatch> {
        let decoded_window = self.submitted.resolve()?;
        let delta = extract_delta_from_padded_with_offset_snapshot(
            &decoded_window,
            &self.previous_audio_lengths,
            &self.semantic_lengths,
            self.audio_offset_frames,
            self.upsample_factor,
        )?;
        self.runtime.decoded_padded_to_pcm_batch(&delta)
    }
}
