use obfstr::obfstr;

use super::{DramFlow, GroupId, RawChannel, Subgroup, dcs_flow, dram_flow, strip_die_prefix};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Channel {
    EnergyRail(Rail),
    CpuCluster(Cluster),
    GpuState,
    AneBandwidth,
    DramBytes(DramFlow),
    DramHistogram(DramFlow),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rail {
    Cpu,
    Gpu,
    Ane,
    Ram,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cluster {
    Efficiency,
    Performance,
}

impl Channel {
    pub(crate) fn classify(channel: &RawChannel) -> Option<Channel> {
        match channel.group {
            GroupId::EnergyModel => Rail::classify(&channel.name).map(Channel::EnergyRail),
            GroupId::CpuStats => (Subgroup::classify(&channel.subgroup) == Subgroup::CpuCorePerformanceStates)
                .then(|| Cluster::classify(&channel.name))
                .flatten()
                .map(Channel::CpuCluster),
            GroupId::GpuStats => (Subgroup::classify(&channel.subgroup) == Subgroup::GpuPerformanceStates
                && channel.name == obfstr!("GPUPH"))
            .then_some(Channel::GpuState),
            GroupId::Pmp => match Subgroup::classify(&channel.subgroup) {
                Subgroup::Floor if channel.name == obfstr!("ANE-AF-BW") || channel.name == obfstr!("ANE-DCS-BW") => {
                    Some(Channel::AneBandwidth)
                },
                Subgroup::DramBandwidth => dram_flow(&channel.name).map(Channel::DramHistogram),
                _ => None,
            },
            GroupId::AmcStats => dcs_flow(strip_die_prefix(&channel.name)).map(Channel::DramBytes),
            GroupId::Other => None,
        }
    }
}

impl Rail {
    fn classify(name: &str) -> Option<Rail> {
        if name == obfstr!("GPU Energy") {
            Some(Rail::Gpu)
        } else if name.ends_with(obfstr!("CPU Energy")) {
            Some(Rail::Cpu)
        } else if name.starts_with(obfstr!("ANE")) {
            Some(Rail::Ane)
        } else if name.starts_with(obfstr!("DRAM"))
            || name.starts_with(obfstr!("DCS"))
            || name.starts_with(obfstr!("AMCC"))
        {
            Some(Rail::Ram)
        } else {
            None
        }
    }
}

impl Cluster {
    fn classify(name: &str) -> Option<Cluster> {
        if name.starts_with(obfstr!("PCPU")) {
            Some(Cluster::Performance)
        } else if name.starts_with(obfstr!("ECPU")) || name.starts_with(obfstr!("MCPU")) {
            Some(Cluster::Efficiency)
        } else {
            None
        }
    }
}
