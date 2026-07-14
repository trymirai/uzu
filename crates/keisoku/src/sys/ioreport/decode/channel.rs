use obfstr::obfstr;

use super::{GroupId, RawChannel, Subgroup, naming};
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
                Subgroup::DramBandwidth => naming::read_write_flow(&channel.name).map(Channel::DramBytes),
                Subgroup::DcsBandwidth if channel.name.starts_with(obfstr!("AMCC")) => {
                    naming::read_write_flow(&channel.name).map(Channel::DramHistogram)
                },
                _ => None,
            },
            GroupId::AmcStats => naming::dcs_flow(naming::strip_die_prefix(&channel.name)).map(Channel::DramBytes),
            GroupId::Other => None,
        }
    }
}
