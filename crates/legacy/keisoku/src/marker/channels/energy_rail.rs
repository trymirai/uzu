use core::marker::PhantomData;

use super::super::{channel_set::IntervalChannel, typelist::ChannelMetric};
use crate::{
    sys::ioreport::{
        IoReportGroups,
        decode::{Channel, RawChannel, energy_joules},
        kinds::RailKind,
    },
    units::Joules,
};

pub struct EnergyRail<R>(PhantomData<R>);

impl<R: RailKind> ChannelMetric for EnergyRail<R> {
    type Value = Joules;
    const TYPE_BIT: u128 = R::TYPE_BIT;
}

impl<R: RailKind> IntervalChannel for EnergyRail<R> {
    const CHANNEL: Channel = Channel::EnergyRail(R::RAIL);
    const GROUP: IoReportGroups = IoReportGroups::ENERGY_MODEL;

    fn default_value() -> Self::Value {
        Joules(0.0)
    }

    fn fold(
        value: &mut Self::Value,
        raw: &RawChannel,
    ) {
        *value = Joules(value.value() + energy_joules(raw.integer_value, raw.unit) as f32);
    }
}
