use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioDecodeStreamingMode {
    IncrementalStateful,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AudioDecodeStepStats {
    pub input_frames: usize,
    pub total_semantic_frames: usize,
    pub decoded_window_start_frame: usize,
    pub decoded_window_frames: usize,
}

/// Ring-buffer backed stream state for incremental audio decoding.
///
/// Token rows are stored in fixed-capacity circular buffers so that evicting
/// old frames is O(1) (just advance the head pointer) instead of O(N) via
/// `Vec::drain`.
#[derive(Debug, Clone)]
pub struct AudioDecodeStreamState {
    pub(super) batch_size: usize,
    pub(super) codebooks: usize,
    retained_capacity: usize,
    stored_frame_start: usize,
    stored_frames: usize,
    total_frames_generated: usize,
    row_tokens: Vec<Vec<u32>>,
    flattened_tokens: Vec<u32>,
    window_lengths: Vec<usize>,
    pub(super) semantic_lengths: Vec<usize>,
    pub(super) emitted_semantic_lengths: Vec<usize>,
    pub(super) emitted_audio_lengths: Vec<usize>,
    last_step_stats: AudioDecodeStepStats,
}

impl AudioDecodeStreamState {
    pub(super) fn new(
        batch_size: usize,
        codebooks: usize,
        max_workspace_frames: usize,
    ) -> AudioResult<Self> {
        if batch_size == 0 {
            return Err(AudioError::Runtime("stream state batch_size must be > 0".to_string()));
        }
        if codebooks == 0 {
            return Err(AudioError::Runtime("stream state codebooks must be > 0".to_string()));
        }
        if max_workspace_frames == 0 {
            return Err(AudioError::Runtime("stream state max_workspace_frames must be > 0".to_string()));
        }

        let row_count = batch_size
            .checked_mul(codebooks)
            .ok_or(AudioError::Runtime("stream state row count overflow".to_string()))?;
        // Double the workspace for ring capacity so we have room for context
        // overlap without immediate eviction.
        let retained_capacity = max_workspace_frames
            .checked_mul(2)
            .ok_or(AudioError::Runtime("stream retained capacity overflow".to_string()))?;
        let mut row_tokens = Vec::with_capacity(row_count);
        for _ in 0..row_count {
            row_tokens.push(vec![0u32; retained_capacity]);
        }

        Ok(Self {
            batch_size,
            codebooks,
            retained_capacity,
            stored_frame_start: 0,
            stored_frames: 0,
            total_frames_generated: 0,
            row_tokens,
            flattened_tokens: Vec::with_capacity(
                row_count
                    .checked_mul(max_workspace_frames)
                    .ok_or(AudioError::Runtime("stream token capacity overflow".to_string()))?,
            ),
            window_lengths: Vec::with_capacity(batch_size),
            semantic_lengths: vec![0; batch_size],
            emitted_semantic_lengths: vec![0; batch_size],
            emitted_audio_lengths: vec![0; batch_size],
            last_step_stats: AudioDecodeStepStats::default(),
        })
    }

    pub(super) fn total_frames(&self) -> usize {
        self.total_frames_generated
    }

    pub fn last_step_stats(&self) -> AudioDecodeStepStats {
        self.last_step_stats
    }

    /// Global frame index one past the last retained frame.
    fn stored_frame_end(&self) -> usize {
        self.stored_frame_start.saturating_add(self.stored_frames)
    }

    /// Physical ring slot for a global frame index.
    #[inline]
    fn slot(&self, global_frame: usize) -> usize {
        global_frame % self.retained_capacity
    }

    pub(super) fn append_delta(
        &mut self,
        delta_tokens: &AudioTokenGrid,
    ) -> AudioResult<()> {
        if delta_tokens.batch_size() != self.batch_size {
            return Err(AudioError::Runtime(format!(
                "stream delta batch mismatch: expected {}, got {}",
                self.batch_size,
                delta_tokens.batch_size()
            )));
        }
        if delta_tokens.codebooks() != self.codebooks {
            return Err(AudioError::Runtime(format!(
                "stream delta codebook mismatch: expected {}, got {}",
                self.codebooks,
                delta_tokens.codebooks()
            )));
        }
        if delta_tokens.frames() == 0 {
            return Ok(());
        }

        let delta_frames = delta_tokens.frames();
        let tokens = delta_tokens.tokens();
        let delta_lengths = delta_tokens.lengths();

        for frame_offset in 0..delta_frames {
            let global_frame = self.stored_frame_start + self.stored_frames;
            // Evict oldest frame if ring is full.
            if self.stored_frames == self.retained_capacity {
                self.stored_frame_start += 1;
                self.stored_frames -= 1;
            }
            let physical = self.slot(global_frame);

            for batch in 0..self.batch_size {
                for codebook in 0..self.codebooks {
                    let row_index = batch * self.codebooks + codebook;
                    let src_index = row_index * delta_frames + frame_offset;
                    self.row_tokens[row_index][physical] = tokens[src_index];
                }
            }
            self.stored_frames += 1;
        }
        self.total_frames_generated = self.total_frames_generated.saturating_add(delta_frames);

        for (length, &delta_len) in self.semantic_lengths.iter_mut().zip(delta_lengths.iter()) {
            *length = length
                .checked_add(delta_len)
                .ok_or(AudioError::Runtime("stream semantic length overflow".to_string()))?;
        }

        Ok(())
    }

    pub(super) fn to_full_grid(&self) -> AudioResult<AudioTokenGrid> {
        if self.stored_frame_start != 0 {
            return Err(AudioError::Runtime(format!(
                "full-grid decode requires retained prefix, but {} frames were evicted",
                self.stored_frame_start
            )));
        }
        let total_frames = self.stored_frames;

        let expected_tokens = self
            .batch_size
            .checked_mul(self.codebooks)
            .and_then(|value| value.checked_mul(total_frames))
            .ok_or(AudioError::Runtime("stream token count overflow".to_string()))?;
        let mut tokens = Vec::with_capacity(expected_tokens);

        for row in &self.row_tokens {
            // When stored_frame_start == 0 and no wrap has occurred, frames
            // 0..total_frames map to physical slots 0..total_frames.
            for frame in 0..total_frames {
                let physical = self.slot(frame);
                tokens.push(row[physical]);
            }
        }
        if tokens.len() != expected_tokens {
            return Err(AudioError::Runtime("stream flattened token count mismatch".to_string()));
        }

        AudioTokenGrid::new(
            tokens.into_boxed_slice(),
            self.batch_size,
            self.codebooks,
            total_frames,
            self.semantic_lengths.clone().into_boxed_slice(),
        )
    }

    pub(super) fn flatten_window(
        &mut self,
        start_frame: usize,
        end_frame: usize,
    ) -> AudioResult<(&[u32], &[usize], usize)> {
        if end_frame < start_frame {
            return Err(AudioError::Runtime(format!(
                "invalid stream window: start_frame={start_frame}, end_frame={end_frame}"
            )));
        }
        let total_frames = self.total_frames();
        if end_frame > total_frames {
            return Err(AudioError::Runtime(format!(
                "stream window end exceeds token frames: end={end_frame}, total={total_frames}"
            )));
        }
        if start_frame < self.stored_frame_start || end_frame > self.stored_frame_end() {
            return Err(AudioError::Runtime(format!(
                "stream window [{start_frame}, {end_frame}) exceeds retained workspace [{}, {})",
                self.stored_frame_start,
                self.stored_frame_end()
            )));
        }

        let window_frames = end_frame.saturating_sub(start_frame);
        let row_count = self
            .batch_size
            .checked_mul(self.codebooks)
            .ok_or(AudioError::Runtime("stream row count overflow".to_string()))?;
        let required_capacity = row_count
            .checked_mul(window_frames)
            .ok_or(AudioError::Runtime("stream window token capacity overflow".to_string()))?;

        // Grow the scratch buffer if needed (only happens once in practice).
        if self.flattened_tokens.capacity() < required_capacity {
            self.flattened_tokens.reserve(required_capacity - self.flattened_tokens.capacity());
        }
        self.flattened_tokens.clear();

        // Read from ring in logical order via modulo indexing.
        for row in &self.row_tokens {
            for global_frame in start_frame..end_frame {
                let physical = self.slot(global_frame);
                self.flattened_tokens.push(row[physical]);
            }
        }

        self.window_lengths.clear();
        for &length in &self.semantic_lengths {
            self.window_lengths.push(length.saturating_sub(start_frame));
        }

        Ok((&self.flattened_tokens, &self.window_lengths, window_frames))
    }

    pub(super) fn extract_delta_from_padded_with_offset(
        &mut self,
        full_padded: &DecodedPaddedAudio,
        audio_offset_frames: usize,
        upsample_factor: usize,
    ) -> AudioResult<DecodedPaddedAudio> {
        let delta = extract_delta_from_padded_with_offset_snapshot(
            full_padded,
            &self.emitted_audio_lengths,
            &self.semantic_lengths,
            audio_offset_frames,
            upsample_factor,
        )?;
        for index in 0..self.semantic_lengths.len() {
            let semantic_length = self.semantic_lengths[index];
            let full_audio_length = semantic_length
                .checked_mul(upsample_factor)
                .ok_or(AudioError::Runtime("stream audio length overflow".to_string()))?;
            self.emitted_audio_lengths[index] = full_audio_length;
            self.emitted_semantic_lengths[index] = semantic_length;
        }
        Ok(delta)
    }

    pub(super) fn mark_submitted_audio_window(
        &mut self,
        semantic_lengths: &[usize],
        upsample_factor: usize,
    ) -> AudioResult<()> {
        if semantic_lengths.len() != self.batch_size {
            return Err(AudioError::Runtime(format!(
                "stream semantic lengths mismatch: expected {}, got {}",
                self.batch_size,
                semantic_lengths.len()
            )));
        }
        for (index, &semantic_length) in semantic_lengths.iter().enumerate() {
            let full_audio_length = semantic_length
                .checked_mul(upsample_factor)
                .ok_or(AudioError::Runtime("stream audio length overflow".to_string()))?;
            self.emitted_audio_lengths[index] = full_audio_length;
            self.emitted_semantic_lengths[index] = semantic_length;
        }
        Ok(())
    }

    pub(super) fn record_last_step_stats(
        &mut self,
        input_frames: usize,
        decoded_window_start_frame: usize,
        decoded_window_frames: usize,
    ) {
        self.last_step_stats = AudioDecodeStepStats {
            input_frames,
            total_semantic_frames: self.total_frames_generated,
            decoded_window_start_frame,
            decoded_window_frames,
        };
    }
}

pub(super) fn extract_delta_from_padded_with_offset_snapshot(
    full_padded: &DecodedPaddedAudio,
    previous_audio_lengths: &[usize],
    semantic_lengths: &[usize],
    audio_offset_frames: usize,
    upsample_factor: usize,
) -> AudioResult<DecodedPaddedAudio> {
    if full_padded.lengths.len() != semantic_lengths.len() || previous_audio_lengths.len() != semantic_lengths.len() {
        return Err(AudioError::Runtime(format!(
            "stream decoded batch mismatch: full_lengths={}, previous_lengths={}, semantic_lengths={}",
            full_padded.lengths.len(),
            previous_audio_lengths.len(),
            semantic_lengths.len()
        )));
    }
    if upsample_factor == 0 {
        return Err(AudioError::Runtime("stream upsample_factor must be > 0".to_string()));
    }

    let batch_size = semantic_lengths.len();
    let channels = full_padded.channels;
    let decoded_frames = full_padded.frames;
    let mut delta_lengths = vec![0usize; batch_size];
    let mut max_delta_frames = 0usize;
    let mut local_delta_ranges = vec![(0usize, 0usize); batch_size];

    for batch in 0..batch_size {
        let full_audio_length = semantic_lengths[batch]
            .checked_mul(upsample_factor)
            .ok_or(AudioError::Runtime("stream audio length overflow".to_string()))?;
        let previous_audio_length = previous_audio_lengths[batch].min(full_audio_length);
        let decoded_batch_length = full_padded.lengths[batch];
        let target_end_local = full_audio_length.saturating_sub(audio_offset_frames).min(decoded_batch_length);
        let target_start_local = previous_audio_length.saturating_sub(audio_offset_frames).min(target_end_local);
        let delta = target_end_local.saturating_sub(target_start_local);
        local_delta_ranges[batch] = (target_start_local, target_end_local);
        delta_lengths[batch] = delta;
        max_delta_frames = max_delta_frames.max(delta);
    }

    let padded_len = checked_product(&[batch_size, channels, max_delta_frames])?;
    let mut delta_padded = vec![0.0_f32; padded_len];
    for (batch, &(start_local, end_local)) in local_delta_ranges.iter().enumerate() {
        let delta_len = end_local.saturating_sub(start_local);
        if delta_len == 0 {
            continue;
        }
        for channel in 0..channels {
            let src_start = (batch * channels + channel)
                .checked_mul(decoded_frames)
                .and_then(|value| value.checked_add(start_local))
                .ok_or(AudioError::Runtime("stream padded source offset overflow".to_string()))?;
            let src_end = src_start
                .checked_add(delta_len)
                .ok_or(AudioError::Runtime("stream padded source offset overflow".to_string()))?;
            let dst_start = (batch * channels + channel)
                .checked_mul(max_delta_frames)
                .ok_or(AudioError::Runtime("stream padded destination offset overflow".to_string()))?;
            let dst_end = dst_start
                .checked_add(delta_len)
                .ok_or(AudioError::Runtime("stream padded destination offset overflow".to_string()))?;
            delta_padded[dst_start..dst_end].copy_from_slice(&full_padded.samples[src_start..src_end]);
        }
    }

    Ok(DecodedPaddedAudio {
        samples: delta_padded,
        channels,
        frames: max_delta_frames,
        lengths: delta_lengths,
    })
}

pub(super) fn pack_pcm_to_padded(
    pcm: &AudioPcmBatch,
    expected_channels: usize,
) -> AudioResult<(Vec<f32>, Vec<usize>, Vec<i32>, usize)> {
    if pcm.channels() != expected_channels {
        return Err(AudioError::Runtime(format!(
            "pcm channel mismatch: expected {expected_channels}, got {}",
            pcm.channels()
        )));
    }

    let lengths = pcm.lengths().to_vec();
    let frames = lengths.iter().copied().max().unwrap_or(0);
    let lengths_i32 = convert_lengths_to_i32(&lengths, frames)?;
    let padded_len = checked_product(&[pcm.batch_size(), expected_channels, frames])?;
    let mut padded = vec![0.0_f32; padded_len];

    let mut src_offset = 0usize;
    let samples = pcm.samples();
    for (batch, &frame_count) in lengths.iter().enumerate() {
        let sample_count = frame_count
            .checked_mul(expected_channels)
            .ok_or(AudioError::Runtime("pcm frame-count overflow".to_string()))?;
        let src_end =
            src_offset.checked_add(sample_count).ok_or(AudioError::Runtime("pcm indexing overflow".to_string()))?;
        if src_end > samples.len() {
            return Err(AudioError::Runtime("pcm indexing out of bounds".to_string()));
        }

        for frame in 0..frame_count {
            let src_frame_base = src_offset + frame * expected_channels;
            for channel in 0..expected_channels {
                let dst_index = (batch * expected_channels + channel) * frames + frame;
                padded[dst_index] = samples[src_frame_base + channel];
            }
        }

        src_offset = src_end;
    }

    Ok((padded, lengths, lengths_i32, frames))
}

pub(super) fn unpack_padded_to_pcm(
    padded: &[f32],
    batch_size: usize,
    channels: usize,
    frames: usize,
    lengths: &[usize],
) -> AudioResult<Vec<f32>> {
    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }

    let expected_padded = checked_product(&[batch_size, channels, frames])?;
    if padded.len() != expected_padded {
        return Err(AudioError::InvalidPcmShape {
            expected_samples: expected_padded,
            actual_samples: padded.len(),
        });
    }

    let total_frames = lengths.iter().sum::<usize>();
    let packed_len = checked_product(&[total_frames, channels])?;
    let mut packed = vec![0.0_f32; packed_len];
    let mut dst_offset = 0usize;

    for (batch, &frame_count) in lengths.iter().enumerate() {
        if frame_count > frames {
            return Err(AudioError::InvalidTokenLengthValue {
                length: frame_count,
                frames,
            });
        }

        for frame in 0..frame_count {
            for channel in 0..channels {
                let src_index = (batch * channels + channel) * frames + frame;
                let dst_index = dst_offset + frame * channels + channel;
                packed[dst_index] = padded[src_index];
            }
        }
        dst_offset = dst_offset
            .checked_add(
                frame_count
                    .checked_mul(channels)
                    .ok_or(AudioError::Runtime("pcm destination indexing overflow".to_string()))?,
            )
            .ok_or(AudioError::Runtime("pcm destination indexing overflow".to_string()))?;
    }

    Ok(packed)
}
