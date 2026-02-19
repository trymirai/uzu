use crate::audio::{AudioResult, AudioTokenGrid, AudioTokenPacking, AudioTokenSpace};

#[derive(Debug, Clone, Copy)]
pub struct AudioModelTokenMapper {
    token_space: AudioTokenSpace,
}

impl AudioModelTokenMapper {
    pub fn new(token_space: AudioTokenSpace) -> Self {
        Self {
            token_space,
        }
    }

    pub fn token_space(&self) -> AudioTokenSpace {
        self.token_space
    }

    pub fn map_codec_grid_to_model_tokens(
        &self,
        grid: &AudioTokenGrid,
    ) -> AudioResult<Box<[u64]>> {
        grid.tokens()
            .iter()
            .map(|&token| self.token_space.try_map_codec_to_model(token))
            .collect::<AudioResult<Vec<_>>>()
            .map(Vec::into_boxed_slice)
    }

    pub fn map_model_tokens_to_codec_grid(
        &self,
        model_tokens: &[u64],
        batch_size: usize,
        codebooks: usize,
        frames: usize,
        lengths: Box<[usize]>,
        packing: AudioTokenPacking,
    ) -> AudioResult<AudioTokenGrid> {
        let codec_tokens = model_tokens
            .iter()
            .map(|&token| self.token_space.map_model_to_codec(token))
            .collect::<AudioResult<Vec<_>>>()?;

        AudioTokenGrid::new(codec_tokens.into_boxed_slice(), batch_size, codebooks, frames, lengths, packing)
    }
}
