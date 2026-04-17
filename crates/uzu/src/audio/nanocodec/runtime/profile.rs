use std::sync::mpsc::Receiver;

use super::*;
use crate::{array::allocation_as_slice, backends::common::Allocation};

pub struct SubmittedDecodedPaddedAudio<B: Backend> {
    pub(in crate::audio::nanocodec::runtime) output: Allocation<B>,
    pub(in crate::audio::nanocodec::runtime) data_type: DataType,
    pub(in crate::audio::nanocodec::runtime) channels: usize,
    pub(in crate::audio::nanocodec::runtime) frames: usize,
    pub(in crate::audio::nanocodec::runtime) lengths: Vec<usize>,
    pub(in crate::audio::nanocodec::runtime) final_command_buffer: Option<Pending<B>>,
    pub(in crate::audio::nanocodec::runtime) completion_notification: Option<Receiver<()>>,
}

impl<B: Backend> SubmittedDecodedPaddedAudio<B> {
    pub(crate) fn is_complete(&self) -> bool {
        use std::sync::mpsc::TryRecvError;

        self.completion_notification.as_ref().is_none_or(|notification| match notification.try_recv() {
            Ok(()) | Err(TryRecvError::Disconnected) => true,
            Err(TryRecvError::Empty) => false,
        })
    }

    pub(in crate::audio::nanocodec::runtime) fn resolve(mut self) -> AudioResult<DecodedPaddedAudio> {
        if let Some(command_buffer) = self.final_command_buffer.take() {
            command_buffer.wait_until_completed().map_err(|err| {
                AudioError::Runtime(format!("failed to wait for FishAudio decoder command buffer: {err}"))
            })?;
        }
        let samples: Vec<f32> = match self.data_type {
            DataType::F32 => allocation_as_slice::<f32, B>(&self.output).to_vec(),
            DataType::F16 => allocation_as_slice::<half::f16, B>(&self.output).iter().map(|&v| f32::from(v)).collect(),
            DataType::BF16 => {
                allocation_as_slice::<half::bf16, B>(&self.output).iter().map(|&v| f32::from(v)).collect()
            },
            dt => return Err(AudioError::Runtime(format!("unsupported vocoder output dtype: {dt:?}"))),
        };
        Ok(DecodedPaddedAudio {
            samples,
            channels: self.channels,
            frames: self.frames,
            lengths: self.lengths,
        })
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
