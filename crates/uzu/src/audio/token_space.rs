use std::num::NonZeroU32;

use super::{AudioError, AudioResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioTokenSpace {
    offset: u64,
    cardinality: NonZeroU32,
}

impl AudioTokenSpace {
    pub fn new(
        offset: u64,
        cardinality: u32,
    ) -> AudioResult<Self> {
        let cardinality = NonZeroU32::new(cardinality).ok_or(AudioError::InvalidTokenCardinality)?;

        Ok(Self {
            offset,
            cardinality,
        })
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn cardinality(&self) -> u32 {
        self.cardinality.get()
    }

    pub fn range_start(&self) -> u64 {
        self.offset
    }

    pub fn range_end(&self) -> u64 {
        self.offset + self.cardinality.get() as u64 - 1
    }

    pub fn map_codec_to_model(
        &self,
        codec_token: u32,
    ) -> u64 {
        self.try_map_codec_to_model(codec_token).expect("codec token outside configured audio token space")
    }

    pub fn try_map_codec_to_model(
        &self,
        codec_token: u32,
    ) -> AudioResult<u64> {
        if codec_token >= self.cardinality.get() {
            return Err(AudioError::InvalidCodecToken {
                token: codec_token,
                cardinality: self.cardinality.get(),
            });
        }

        Ok(self.offset + codec_token as u64)
    }

    pub fn map_model_to_codec(
        &self,
        model_token: u64,
    ) -> AudioResult<u32> {
        let start = self.range_start();
        let end = self.range_end();

        if model_token < start || model_token > end {
            return Err(AudioError::InvalidModelToken {
                token: model_token,
                range_start: start,
                range_end: end,
            });
        }

        Ok((model_token - start) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::{AudioError, AudioTokenSpace};

    #[test]
    fn mapping_roundtrip_is_lossless() {
        let space = AudioTokenSpace::new(32_000, 1_024).expect("valid token space");
        for codec in [0u32, 7, 511, 1023] {
            let model = space.map_codec_to_model(codec);
            let restored = space.map_model_to_codec(model).expect("valid roundtrip token");
            assert_eq!(restored, codec);
        }
    }

    #[test]
    fn out_of_range_model_token_is_rejected() {
        let space = AudioTokenSpace::new(100, 8).expect("valid token space");
        let error = space.map_model_to_codec(99).expect_err("lower out-of-range token should fail");

        assert!(matches!(
            error,
            AudioError::InvalidModelToken {
                token: 99,
                range_start: 100,
                range_end: 107
            }
        ));
    }
}
