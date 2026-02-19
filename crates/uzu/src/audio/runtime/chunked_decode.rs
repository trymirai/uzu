use std::sync::Arc;

use crate::audio::{AudioCodecRuntime, AudioError, AudioPcmBatch, AudioResult, AudioTokenGrid, AudioTokenPacking};

#[derive(Clone)]
pub struct ChunkedAudioDecoder {
    runtime: Arc<dyn AudioCodecRuntime>,
    chunk_frames: usize,
}

impl ChunkedAudioDecoder {
    pub fn new(
        runtime: Arc<dyn AudioCodecRuntime>,
        chunk_frames: usize,
    ) -> AudioResult<Self> {
        if chunk_frames == 0 {
            return Err(AudioError::InvalidChunkFrames);
        }

        Ok(Self {
            runtime,
            chunk_frames,
        })
    }

    pub fn chunk_frames(&self) -> usize {
        self.chunk_frames
    }

    pub fn decode_chunked(
        &self,
        tokens: &AudioTokenGrid,
    ) -> AudioResult<AudioPcmBatch> {
        if tokens.frames() <= self.chunk_frames {
            return self.runtime.decode(tokens);
        }

        let frame_major_tokens = tokens.to_packing(AudioTokenPacking::FrameMajor);
        let batch_size = frame_major_tokens.batch_size();
        let mut sample_rate = None;
        let mut channels = None;
        let mut merged_lengths = vec![0usize; batch_size];
        let mut merged_samples = vec![Vec::<f32>::new(); batch_size];
        let mut decoded_chunks = 0usize;

        for chunk_start in (0..frame_major_tokens.frames()).step_by(self.chunk_frames) {
            let chunk_end = (chunk_start + self.chunk_frames).min(frame_major_tokens.frames());
            let prefix_grid = slice_frame_range(&frame_major_tokens, 0, chunk_end)?;
            if prefix_grid.lengths().iter().all(|&length| length == 0) {
                continue;
            }

            // Decode progressively larger prefixes and append only the new PCM tail.
            // This preserves temporal context for causal/stateful decoders across chunk boundaries.
            let decoded = self.runtime.decode(&prefix_grid)?;
            if decoded.batch_size() != batch_size {
                return Err(AudioError::Runtime(format!(
                    "audio runtime returned batch_size {}, expected {}",
                    decoded.batch_size(),
                    batch_size
                )));
            }

            match (sample_rate, channels) {
                (None, None) => {
                    sample_rate = Some(decoded.sample_rate());
                    channels = Some(decoded.channels());
                },
                (Some(expected_sample_rate), Some(expected_channels)) => {
                    if decoded.sample_rate() != expected_sample_rate || decoded.channels() != expected_channels {
                        return Err(AudioError::Runtime(
                            "audio runtime changed PCM format between decode chunks".into(),
                        ));
                    }
                },
                _ => unreachable!("sample_rate and channels are set together"),
            }

            let chunk_channels = decoded.channels();
            let mut sample_offset = 0usize;
            for (batch_index, &decoded_length) in decoded.lengths().iter().enumerate() {
                let sample_count = decoded_length * chunk_channels;
                let next_offset = sample_offset + sample_count;
                let batch_samples = decoded
                    .samples()
                    .get(sample_offset..next_offset)
                    .ok_or_else(|| AudioError::Runtime("audio runtime returned malformed chunk PCM layout".into()))?;

                let already_emitted = merged_lengths[batch_index];
                if decoded_length < already_emitted {
                    return Err(AudioError::Runtime(format!(
                        "audio runtime returned non-monotonic PCM length for batch {batch_index}: {decoded_length} < {already_emitted}"
                    )));
                }

                let already_emitted_samples = already_emitted * chunk_channels;
                let new_samples = batch_samples
                    .get(already_emitted_samples..)
                    .ok_or_else(|| AudioError::Runtime("audio runtime returned malformed chunk PCM layout".into()))?;

                merged_samples[batch_index].extend_from_slice(new_samples);
                merged_lengths[batch_index] = decoded_length;
                sample_offset = next_offset;
            }

            if sample_offset != decoded.samples().len() {
                return Err(AudioError::Runtime("audio runtime returned extra PCM data for decode chunk".into()));
            }

            decoded_chunks += 1;
        }

        if decoded_chunks == 0 {
            return self.runtime.decode(tokens);
        }

        let sample_rate = sample_rate.expect("at least one decoded chunk sets sample_rate");
        let channels = channels.expect("at least one decoded chunk sets channels");
        let total_samples = merged_lengths.iter().sum::<usize>() * channels;

        let mut samples = Vec::with_capacity(total_samples);
        for batch_samples in merged_samples {
            samples.extend(batch_samples);
        }

        AudioPcmBatch::new(samples.into_boxed_slice(), sample_rate, channels, merged_lengths.into_boxed_slice())
    }
}

fn slice_frame_range(
    source: &AudioTokenGrid,
    start_frame: usize,
    end_frame: usize,
) -> AudioResult<AudioTokenGrid> {
    debug_assert!(source.packing() == AudioTokenPacking::FrameMajor);
    debug_assert!(start_frame <= end_frame);

    let chunk_frames = end_frame.saturating_sub(start_frame);
    let mut chunk_tokens = vec![0u32; source.batch_size() * source.codebooks() * chunk_frames];

    for batch in 0..source.batch_size() {
        for frame_in_chunk in 0..chunk_frames {
            let source_frame = start_frame + frame_in_chunk;
            for codebook in 0..source.codebooks() {
                let source_index = ((batch * source.frames() + source_frame) * source.codebooks()) + codebook;
                let chunk_index = ((batch * chunk_frames + frame_in_chunk) * source.codebooks()) + codebook;
                chunk_tokens[chunk_index] = source.tokens()[source_index];
            }
        }
    }

    let chunk_lengths = source
        .lengths()
        .iter()
        .map(|&length| length.saturating_sub(start_frame).min(chunk_frames))
        .collect::<Vec<_>>()
        .into_boxed_slice();

    AudioTokenGrid::new(
        chunk_tokens.into_boxed_slice(),
        source.batch_size(),
        source.codebooks(),
        chunk_frames,
        chunk_lengths,
        AudioTokenPacking::FrameMajor,
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::audio::{AudioCodecRuntime, AudioPcmBatch, AudioResult};

    use super::{AudioError, AudioTokenGrid, AudioTokenPacking, ChunkedAudioDecoder};

    struct MockRuntime;

    impl AudioCodecRuntime for MockRuntime {
        fn encode(
            &self,
            _pcm: &AudioPcmBatch,
        ) -> AudioResult<AudioTokenGrid> {
            Err(AudioError::Runtime("encode is not used in chunked decode test".into()))
        }

        fn decode(
            &self,
            tokens: &AudioTokenGrid,
        ) -> AudioResult<AudioPcmBatch> {
            let tokens = tokens.to_packing(AudioTokenPacking::FrameMajor);
            let mut out = Vec::new();

            for batch in 0..tokens.batch_size() {
                let length = tokens.lengths()[batch];
                for frame in 0..length {
                    out.push(tokens.get(batch, 0, frame) as f32);
                }
            }

            AudioPcmBatch::new(out.into_boxed_slice(), 24_000, 1, tokens.lengths().to_vec().into_boxed_slice())
        }
    }

    struct CausalContextRuntime;

    impl AudioCodecRuntime for CausalContextRuntime {
        fn encode(
            &self,
            _pcm: &AudioPcmBatch,
        ) -> AudioResult<AudioTokenGrid> {
            Err(AudioError::Runtime("encode is not used in chunked decode test".into()))
        }

        fn decode(
            &self,
            tokens: &AudioTokenGrid,
        ) -> AudioResult<AudioPcmBatch> {
            let tokens = tokens.to_packing(AudioTokenPacking::FrameMajor);
            let mut out = Vec::new();

            for batch in 0..tokens.batch_size() {
                let length = tokens.lengths()[batch];
                for frame in 0..length {
                    let current = tokens.get(batch, 0, frame) as f32;
                    let previous = if frame == 0 {
                        0.0
                    } else {
                        tokens.get(batch, 0, frame - 1) as f32
                    };
                    out.push(current + previous);
                }
            }

            AudioPcmBatch::new(out.into_boxed_slice(), 24_000, 1, tokens.lengths().to_vec().into_boxed_slice())
        }
    }

    #[test]
    fn chunked_decode_matches_full_decode() {
        let grid = AudioTokenGrid::new(
            vec![
                1, 10, // b0 f0
                2, 11, // b0 f1
                3, 12, // b0 f2
                4, 13, // b0 f3
                5, 14, // b0 f4
            ]
            .into_boxed_slice(),
            1,
            2,
            5,
            vec![5].into_boxed_slice(),
            AudioTokenPacking::FrameMajor,
        )
        .expect("valid grid");

        let runtime = Arc::new(MockRuntime);
        let chunked = ChunkedAudioDecoder::new(runtime.clone(), 2).expect("valid decoder");
        let chunked_pcm = chunked.decode_chunked(&grid).expect("chunked decode should work");
        let full_pcm = runtime.decode(&grid).expect("full decode should work");

        assert_eq!(chunked_pcm, full_pcm);
    }

    #[test]
    fn chunked_decode_preserves_causal_context_across_boundaries() {
        let grid = AudioTokenGrid::new(
            vec![
                1, // b0 f0
                2, // b0 f1
                3, // b0 f2
                4, // b0 f3
                5, // b0 f4
            ]
            .into_boxed_slice(),
            1,
            1,
            5,
            vec![5].into_boxed_slice(),
            AudioTokenPacking::FrameMajor,
        )
        .expect("valid grid");

        let runtime = Arc::new(CausalContextRuntime);
        let chunked = ChunkedAudioDecoder::new(runtime.clone(), 2).expect("valid decoder");
        let chunked_pcm = chunked.decode_chunked(&grid).expect("chunked decode should work");
        let full_pcm = runtime.decode(&grid).expect("full decode should work");

        assert_eq!(chunked_pcm, full_pcm);
    }
}
