use rodio::{MixerDeviceSink, Player};

pub struct PlayerContext {
    pub _output: MixerDeviceSink,
    pub player: Player,
}
