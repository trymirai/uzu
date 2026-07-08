use super::{Channel, ChannelFold, FrequencyTables, RawChannel, calculate_frequency};

#[derive(Default, Clone, Copy)]
pub(crate) struct GpuResidency {
    pub(crate) frequency: u32,
    pub(crate) usage: f32,
}

impl ChannelFold for GpuResidency {
    fn fold(
        &mut self,
        channel: Channel,
        raw: &RawChannel,
        frequencies: Option<&FrequencyTables<'_>>,
    ) {
        let (Channel::GpuState, Some(freq)) = (channel, frequencies) else {
            return;
        };
        if freq.gpu.len() <= 1 {
            return;
        }
        let (frequency, usage) = calculate_frequency(&raw.states, &freq.gpu[1..]);
        self.frequency = frequency;
        self.usage = usage;
    }
}
