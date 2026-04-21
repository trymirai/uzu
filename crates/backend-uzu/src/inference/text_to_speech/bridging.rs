use shoji::types::session::text_to_speech::PcmBatch as ShojiPcmBatch;

use crate::{audio::AudioPcmBatch, session::types::Input};

pub fn build_input(input: &str) -> Input {
    Input::Text(input.to_string())
}

pub fn build_pcm_batch(pcm: &AudioPcmBatch) -> ShojiPcmBatch {
    ShojiPcmBatch {
        samples: pcm.samples().iter().map(|value| *value as f64).collect(),
        sample_rate: pcm.sample_rate(),
        channels: pcm.channels() as u32,
        lengths: pcm.lengths().iter().map(|value| *value as u32).collect(),
    }
}
