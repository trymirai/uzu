use obfstr::obfstr;

use super::{GroupId, RawChannel, Subgroup, dcs_flow, read_write_flow, strip_die_prefix};
use crate::sys::ioreport::kinds::{DramFlow, Rail};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Channel {
    EnergyRail(Rail),
    AneBandwidth,
    DramBytes(DramFlow),
    DramHistogram(DramFlow),
}

impl Channel {
    pub fn classify(channel: &RawChannel) -> Option<Channel> {
        match channel.group {
            GroupId::EnergyModel => Rail::classify(&channel.name).map(Channel::EnergyRail),
            GroupId::Pmp => match Subgroup::classify(&channel.subgroup) {
                Subgroup::Floor if channel.name == obfstr!("ANE-AF-BW") || channel.name == obfstr!("ANE-DCS-BW") => {
                    Some(Channel::AneBandwidth)
                },
                Subgroup::DramBandwidth => read_write_flow(&channel.name).map(Channel::DramBytes),
                Subgroup::DcsBandwidth if channel.name.starts_with(obfstr!("AMCC")) => {
                    read_write_flow(&channel.name).map(Channel::DramHistogram)
                },
                _ => None,
            },
            GroupId::AmcStats => dcs_flow(strip_die_prefix(&channel.name)).map(Channel::DramBytes),
            GroupId::CpuStats | GroupId::GpuStats | GroupId::Other => None,
        }
    }
}

impl Rail {
    pub(crate) fn classify(name: &str) -> Option<Rail> {
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
