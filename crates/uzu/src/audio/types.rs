use std::num::{NonZeroU32, NonZeroUsize};

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum AudioError {
    #[error("sample_rate must be > 0")]
    InvalidSampleRate,
    #[error("channels must be > 0")]
    InvalidChannelCount,
    #[error("pcm samples shape mismatch: expected {expected_samples} samples, got {actual_samples}")]
    InvalidPcmShape {
        expected_samples: usize,
        actual_samples: usize,
    },
    #[error("token lengths shape mismatch: expected {expected_lengths} lengths, got {actual_lengths}")]
    InvalidTokenLengths {
        expected_lengths: usize,
        actual_lengths: usize,
    },
    #[error("token grid shape mismatch: expected {expected_tokens} tokens, got {actual_tokens}")]
    InvalidTokenShape {
        expected_tokens: usize,
        actual_tokens: usize,
    },
    #[error("token length {length} exceeds frame count {frames}")]
    InvalidTokenLengthValue {
        length: usize,
        frames: usize,
    },
    #[error("audio token cardinality must be > 0")]
    InvalidTokenCardinality,
    #[error("codec token {token} is outside codec range 0..{cardinality}")]
    InvalidCodecToken {
        token: u32,
        cardinality: u32,
    },
    #[error("model token {token} is outside audio model range {range_start}..={range_end}")]
    InvalidModelToken {
        token: u64,
        range_start: u64,
        range_end: u64,
    },
    #[error("audio decode chunk_frames must be > 0")]
    InvalidChunkFrames,
    #[error("tokenizer operation failed: {0}")]
    Tokenizer(String),
    #[error("audio runtime rejected input: {0}")]
    Runtime(String),
}

pub type AudioResult<T> = Result<T, AudioError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AudioTokenPacking {
    #[default]
    FrameMajor,
    CodebookMajor,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioPcmBatch {
    samples: Box<[f32]>,
    sample_rate: NonZeroU32,
    channels: NonZeroUsize,
    lengths: Box<[usize]>,
}

impl AudioPcmBatch {
    pub fn new(
        samples: Box<[f32]>,
        sample_rate: u32,
        channels: usize,
        lengths: Box<[usize]>,
    ) -> AudioResult<Self> {
        let sample_rate = NonZeroU32::new(sample_rate).ok_or(AudioError::InvalidSampleRate)?;
        let channels = NonZeroUsize::new(channels).ok_or(AudioError::InvalidChannelCount)?;
        let expected_samples = lengths.iter().sum::<usize>() * channels.get();
        if samples.len() != expected_samples {
            return Err(AudioError::InvalidPcmShape {
                expected_samples,
                actual_samples: samples.len(),
            });
        }

        Ok(Self {
            samples,
            sample_rate,
            channels,
            lengths,
        })
    }

    pub fn batch_size(&self) -> usize {
        self.lengths.len()
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate.get()
    }

    pub fn channels(&self) -> usize {
        self.channels.get()
    }

    pub fn lengths(&self) -> &[usize] {
        &self.lengths
    }

    pub fn total_frames(&self) -> usize {
        self.lengths.iter().sum()
    }

    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    pub fn into_parts(self) -> (Box<[f32]>, u32, usize, Box<[usize]>) {
        (self.samples, self.sample_rate.get(), self.channels.get(), self.lengths)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioTokenGrid {
    tokens: Box<[u32]>,
    batch_size: usize,
    codebooks: usize,
    frames: usize,
    lengths: Box<[usize]>,
    packing: AudioTokenPacking,
}

impl AudioTokenGrid {
    pub fn new(
        tokens: Box<[u32]>,
        batch_size: usize,
        codebooks: usize,
        frames: usize,
        lengths: Box<[usize]>,
        packing: AudioTokenPacking,
    ) -> AudioResult<Self> {
        if codebooks == 0 {
            return Err(AudioError::InvalidTokenCardinality);
        }
        if lengths.len() != batch_size {
            return Err(AudioError::InvalidTokenLengths {
                expected_lengths: batch_size,
                actual_lengths: lengths.len(),
            });
        }
        for &length in lengths.iter() {
            if length > frames {
                return Err(AudioError::InvalidTokenLengthValue {
                    length,
                    frames,
                });
            }
        }

        let expected_tokens = batch_size.checked_mul(codebooks).and_then(|n| n.checked_mul(frames)).ok_or(
            AudioError::InvalidTokenShape {
                expected_tokens: usize::MAX,
                actual_tokens: tokens.len(),
            },
        )?;

        if tokens.len() != expected_tokens {
            return Err(AudioError::InvalidTokenShape {
                expected_tokens,
                actual_tokens: tokens.len(),
            });
        }

        Ok(Self {
            tokens,
            batch_size,
            codebooks,
            frames,
            lengths,
            packing,
        })
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn codebooks(&self) -> usize {
        self.codebooks
    }

    pub fn frames(&self) -> usize {
        self.frames
    }

    pub fn lengths(&self) -> &[usize] {
        &self.lengths
    }

    pub fn packing(&self) -> AudioTokenPacking {
        self.packing
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn into_parts(self) -> (Box<[u32]>, usize, usize, usize, Box<[usize]>, AudioTokenPacking) {
        (self.tokens, self.batch_size, self.codebooks, self.frames, self.lengths, self.packing)
    }

    pub fn get(
        &self,
        batch: usize,
        codebook: usize,
        frame: usize,
    ) -> u32 {
        let index =
            Self::token_index(self.packing, self.batch_size, self.codebooks, self.frames, batch, codebook, frame);
        self.tokens[index]
    }

    pub fn to_packing(
        &self,
        packing: AudioTokenPacking,
    ) -> Self {
        if packing == self.packing {
            return self.clone();
        }

        let mut tokens = vec![0u32; self.tokens.len()];
        for batch in 0..self.batch_size {
            for frame in 0..self.frames {
                for codebook in 0..self.codebooks {
                    let src_idx = Self::token_index(
                        self.packing,
                        self.batch_size,
                        self.codebooks,
                        self.frames,
                        batch,
                        codebook,
                        frame,
                    );
                    let dst_idx = Self::token_index(
                        packing,
                        self.batch_size,
                        self.codebooks,
                        self.frames,
                        batch,
                        codebook,
                        frame,
                    );
                    tokens[dst_idx] = self.tokens[src_idx];
                }
            }
        }

        Self {
            tokens: tokens.into_boxed_slice(),
            batch_size: self.batch_size,
            codebooks: self.codebooks,
            frames: self.frames,
            lengths: self.lengths.clone(),
            packing,
        }
    }

    fn token_index(
        packing: AudioTokenPacking,
        batch_size: usize,
        codebooks: usize,
        frames: usize,
        batch: usize,
        codebook: usize,
        frame: usize,
    ) -> usize {
        debug_assert!(batch < batch_size);
        debug_assert!(codebook < codebooks);
        debug_assert!(frame < frames);

        match packing {
            AudioTokenPacking::FrameMajor => ((batch * frames + frame) * codebooks) + codebook,
            AudioTokenPacking::CodebookMajor => ((batch * codebooks + codebook) * frames) + frame,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AudioError, AudioTokenGrid, AudioTokenPacking};

    #[test]
    fn packing_conversion_roundtrip_is_lossless() {
        let grid = AudioTokenGrid::new(
            vec![
                0, 1, 2, 3, // b0 f0, f1
                4, 5, 6, 7, // b1 f0, f1
            ]
            .into_boxed_slice(),
            2,
            2,
            2,
            vec![2, 2].into_boxed_slice(),
            AudioTokenPacking::FrameMajor,
        )
        .expect("valid grid");

        let converted = grid.to_packing(AudioTokenPacking::CodebookMajor);
        let restored = converted.to_packing(AudioTokenPacking::FrameMajor);

        assert_eq!(restored, grid);
    }

    #[test]
    fn invalid_grid_shape_is_rejected() {
        let error = AudioTokenGrid::new(
            vec![0, 1, 2].into_boxed_slice(),
            1,
            2,
            2,
            vec![2].into_boxed_slice(),
            AudioTokenPacking::FrameMajor,
        )
        .expect_err("shape mismatch should fail");

        assert!(matches!(
            error,
            AudioError::InvalidTokenShape {
                expected_tokens: 4,
                actual_tokens: 3
            }
        ));
    }
}
