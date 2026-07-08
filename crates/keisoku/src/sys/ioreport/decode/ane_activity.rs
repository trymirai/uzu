use super::{Channel, ChannelFold, FrequencyTables, RawChannel, residency_active_percent};

#[derive(Default, Clone, Copy)]
pub(crate) struct AneActivity {
    pub(crate) active_percent: f32,
}

impl ChannelFold for AneActivity {
    fn fold(
        &mut self,
        channel: Channel,
        raw: &RawChannel,
        _frequencies: Option<&FrequencyTables<'_>>,
    ) {
        let Channel::AneBandwidth = channel else {
            return;
        };
        self.active_percent = self.active_percent.max(residency_active_percent(&raw.states));
    }
}
