use super::{Channel, ChannelFold, DramFlow, FrequencyTables, RawChannel, residency_weighted_gbps};

#[derive(Default, Clone)]
pub(crate) struct DramBandwidth {
    pub(crate) read_bytes: f64,
    pub(crate) write_bytes: f64,
    pub(crate) read_histogram: f32,
    pub(crate) write_histogram: f32,
}

impl ChannelFold for DramBandwidth {
    fn fold(
        &mut self,
        channel: Channel,
        raw: &RawChannel,
        _frequencies: Option<&FrequencyTables<'_>>,
    ) {
        match channel {
            Channel::DramBytes(flow) => {
                let bytes = raw.integer_value as f64;
                if bytes > 0.0 {
                    match flow {
                        DramFlow::Read => self.read_bytes += bytes,
                        DramFlow::Write => self.write_bytes += bytes,
                        DramFlow::Combined => {},
                    }
                }
            },
            Channel::DramHistogram(flow) => {
                let gbps = residency_weighted_gbps(&raw.states);
                match flow {
                    DramFlow::Read => self.read_histogram = self.read_histogram.max(gbps),
                    DramFlow::Write => self.write_histogram = self.write_histogram.max(gbps),
                    DramFlow::Combined => {},
                }
            },
            _ => {},
        }
    }
}
