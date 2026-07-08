use super::{Channel, ChannelFold, Cluster, FrequencyTables, RawChannel, calculate_frequency};

#[derive(Default, Clone)]
pub(crate) struct CpuResidency {
    pub(crate) ecpu: Vec<(u32, f32)>,
    pub(crate) pcpu: Vec<(u32, f32)>,
}

impl ChannelFold for CpuResidency {
    fn fold(
        &mut self,
        channel: Channel,
        raw: &RawChannel,
        frequencies: Option<&FrequencyTables<'_>>,
    ) {
        let (Channel::CpuCluster(cluster), Some(freq)) = (channel, frequencies) else {
            return;
        };
        match cluster {
            Cluster::Performance => self.pcpu.push(calculate_frequency(&raw.states, freq.pcpu)),
            Cluster::Efficiency => self.ecpu.push(calculate_frequency(&raw.states, freq.ecpu)),
        }
    }
}
