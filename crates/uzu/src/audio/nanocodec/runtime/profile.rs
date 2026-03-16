use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NanoCodecFsqRuntimeOptions {
    pub chunked_command_buffers: bool,
    pub micro_flush_min_elements: usize,
}

impl Default for NanoCodecFsqRuntimeOptions {
    fn default() -> Self {
        Self {
            chunked_command_buffers: true,
            micro_flush_min_elements: 8_000_000,
        }
    }
}

pub struct SubmittedDecodedPaddedAudio<B: Backend> {
    pub(in crate::audio::nanocodec::runtime) output: Array<B>,
    pub(in crate::audio::nanocodec::runtime) channels: usize,
    pub(in crate::audio::nanocodec::runtime) frames: usize,
    pub(in crate::audio::nanocodec::runtime) lengths: Vec<usize>,
    pub(in crate::audio::nanocodec::runtime) final_command_buffer: Option<<B::CommandBuffer as CommandBuffer>::Pending>,
}

impl<B: Backend> SubmittedDecodedPaddedAudio<B> {
    pub(in crate::audio::nanocodec::runtime) fn is_ready(&self) -> bool {
        self.final_command_buffer.as_ref().is_none_or(|command_buffer| command_buffer.is_completed())
    }

    pub(in crate::audio::nanocodec::runtime) fn resolve(
        mut self
    ) -> AudioResult<DecodedPaddedAudio> {
        if let Some(command_buffer) = self.final_command_buffer.take() {
            command_buffer.wait_until_completed().map_err(|err| {
                AudioError::Runtime(format!("failed to wait for FishAudio decoder command buffer: {err}"))
            })?;
        }
        let samples = read_array_to_f32_vec(&self.output)?;
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
    pub(crate) fn is_ready(&self) -> bool {
        self.submitted.is_ready()
    }

    pub(crate) fn step_stats(&self) -> AudioDecodeStepStats {
        self.step_stats
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
