mod context;
mod error;

use std::{num::NonZero, sync::Arc};

use context::PlayerContext;
pub use error::PlayerError;
use rodio::{DeviceSinkBuilder, Player as RodioPlayer, Sample, buffer::SamplesBuffer};
use shoji::types::basic::PcmBatch;

#[bindings::export(Class)]
#[derive(Clone)]
pub struct Player {
    context: Arc<PlayerContext>,
}

impl Player {
    pub fn new() -> Result<Self, PlayerError> {
        let mut output = DeviceSinkBuilder::open_default_sink().map_err(|error| PlayerError::RodioError {
            message: error.to_string(),
        })?;
        output.log_on_drop(false);

        let player = RodioPlayer::connect_new(output.mixer());
        Ok(Self {
            context: Arc::new(PlayerContext {
                _output: output,
                player,
            }),
        })
    }
}

#[bindings::export(Implementation)]
impl Player {
    #[bindings::export(Method(Factory))]
    pub async fn create() -> Result<Self, PlayerError> {
        Self::new()
    }

    #[bindings::export(Method)]
    pub fn append_pcm_batch(
        &self,
        batch: PcmBatch,
    ) -> Result<(), PlayerError> {
        let sample_rate = NonZero::new(batch.sample_rate).ok_or_else(|| PlayerError::InvalidPcmBatch {
            message: "Sample rate must be greater than zero".to_string(),
        })?;
        let channels = NonZero::new(u16::try_from(batch.channels).map_err(|_| PlayerError::InvalidPcmBatch {
            message: "Channels must fit into u16".to_string(),
        })?)
        .ok_or_else(|| PlayerError::InvalidPcmBatch {
            message: "Channels must be greater than zero".to_string(),
        })?;
        let channel_count = channels.get() as usize;

        if batch.samples.len() % channel_count != 0 {
            return Err(PlayerError::InvalidPcmBatch {
                message: "Sample count must be divisible by channel count".to_string(),
            });
        }

        let samples = batch.samples.into_iter().map(|sample| sample.clamp(-1.0, 1.0) as Sample).collect::<Vec<_>>();
        self.context.player.append(SamplesBuffer::new(channels, sample_rate, samples));
        self.context.player.play();
        Ok(())
    }

    #[bindings::export(Method)]
    pub fn stop(&self) {
        self.context.player.stop();
    }
}
