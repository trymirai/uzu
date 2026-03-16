use super::*;

pub(super) struct NanoCodecAudioDecoderBackend<B: StructuredDecoderBackend> {
    audio: AudioGenerationContext<B>,
    codec_cardinality: usize,
}

pub(super) struct NanoCodecAudioDecoderStream<B: StructuredDecoderBackend> {
    audio: AudioGenerationContext<B>,
    state: Option<AudioDecodeStreamState>,
}

impl<B: StructuredDecoderBackend> NanoCodecAudioDecoderBackend<B> {
    pub(super) fn new(audio: AudioGenerationContext<B>) -> Self {
        let codec_cardinality = audio.codec_cardinality();
        Self {
            audio,
            codec_cardinality,
        }
    }
}

impl<B: StructuredDecoderBackend> AudioDecoderBackend for NanoCodecAudioDecoderBackend<B> {
    fn codec_cardinality(&self) -> usize {
        self.codec_cardinality
    }

    fn num_codebooks(&self) -> usize {
        self.audio.runtime().config().num_groups()
    }

    fn sample_rate(&self) -> u32 {
        self.audio.sample_rate()
    }

    fn decode(
        &self,
        tokens: &AudioTokenGrid,
    ) -> Result<AudioPcmBatch, Error> {
        self.audio.runtime().decode(tokens).map_err(Error::from)
    }

    fn begin_stream(
        &self,
        batch_size: usize,
        codebooks: usize,
        mode: AudioDecodeStreamingMode,
        max_workspace_frames: usize,
    ) -> Result<Box<dyn AudioDecoderStreamBackend>, Error> {
        let state = self
            .audio
            .runtime()
            .begin_decode_stream_with_options(batch_size, codebooks, mode, max_workspace_frames)
            .map_err(Error::from)?;
        Ok(Box::new(NanoCodecAudioDecoderStream {
            audio: self.audio.clone(),
            state: Some(state),
        }))
    }
}

impl<B: StructuredDecoderBackend> AudioDecoderStreamBackend for NanoCodecAudioDecoderStream<B> {
    fn decode_step(
        &mut self,
        new_tokens: &AudioTokenGrid,
        is_final: bool,
    ) -> Result<AudioPcmBatch, Error> {
        let state = self.state.as_mut().ok_or(Error::GenerateFailed)?;
        let decoded = self.audio.runtime().decode_stream_step(state, new_tokens, is_final).map_err(Error::from)?;
        self.audio.runtime().decoded_padded_to_pcm_batch(&decoded).map_err(Error::from)
    }

    fn decode_step_pending(
        &mut self,
        new_tokens: &AudioTokenGrid,
        is_final: bool,
    ) -> Result<Box<dyn PendingAudioChunkBackend>, Error> {
        let state = self.state.as_mut().ok_or(Error::GenerateFailed)?;
        if let Some(pending) =
            self.audio.runtime().submit_decode_stream_step(state, new_tokens, is_final).map_err(Error::from)?
        {
            return Ok(Box::new(NanoCodecPendingAudioChunk {
                inner: Some(pending),
            }));
        }

        let decoded = self.audio.runtime().decode_stream_step(state, new_tokens, is_final).map_err(Error::from)?;
        let pcm = self.audio.runtime().decoded_padded_to_pcm_batch(&decoded).map_err(Error::from)?;
        Ok(Box::new(ImmediatePendingAudioChunk {
            pcm: Some(pcm),
            step_stats: Some(state.last_step_stats()),
        }))
    }

    fn last_step_stats(&self) -> Option<AudioDecodeStepStats> {
        self.state.as_ref().map(AudioDecodeStreamState::last_step_stats)
    }

    fn finish(mut self: Box<Self>) -> Result<(), Error> {
        let state = self.state.take().ok_or(Error::GenerateFailed)?;
        self.audio.runtime().end_decode_stream(state).map_err(Error::from)
    }
}
