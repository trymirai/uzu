use super::{AudioPcmBatch, AudioResult, AudioTokenGrid};

pub trait AudioCodecRuntime {
    fn encode(
        &self,
        pcm: &AudioPcmBatch,
    ) -> AudioResult<AudioTokenGrid>;

    fn decode(
        &self,
        tokens: &AudioTokenGrid,
    ) -> AudioResult<AudioPcmBatch>;
}
