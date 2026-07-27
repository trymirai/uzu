use super::super::{channel_set::IntervalChannel, typelist::ChannelMetric};
use crate::{
    sys::ioreport::{
        IoReportGroups,
        decode::{Channel, RawChannel, residency_active_percent},
    },
    units::Percent,
};

pub struct AneBandwidth;

impl ChannelMetric for AneBandwidth {
    type Value = Percent;
    const TYPE_BIT: u128 = 1 << 4;
}

impl IntervalChannel for AneBandwidth {
    const CHANNEL: Channel = Channel::AneBandwidth;
    const GROUP: IoReportGroups = IoReportGroups::PMP;

    fn default_value() -> Self::Value {
        Percent(0.0)
    }

    fn fold(
        value: &mut Self::Value,
        raw: &RawChannel,
    ) {
        *value = Percent(value.value().max(residency_active_percent(&raw.states)));
    }
}
