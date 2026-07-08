use super::{Channel, ChannelFold, DramFlow, FrequencyTables, RawChannel};

#[derive(Default, Clone)]
pub(crate) struct DramBandwidth {
    pub(crate) read_bytes: f64,
    pub(crate) write_bytes: f64,
}

impl ChannelFold for DramBandwidth {
    fn fold(
        &mut self,
        channel: Channel,
        raw: &RawChannel,
        _frequencies: Option<&FrequencyTables<'_>>,
    ) {
        let Channel::DramBytes(flow) = channel else {
            return;
        };
        let bytes = raw.integer_value as f64;
        if bytes > 0.0 {
            match flow {
                DramFlow::Read => self.read_bytes += bytes,
                DramFlow::Write => self.write_bytes += bytes,
            }
        }
    }
}
