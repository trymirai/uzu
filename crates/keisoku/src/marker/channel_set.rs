use super::{
    Cons as ConsType, Nil as NilType,
    typelist::{ChannelMetric, ChannelSet, Values},
};
use crate::sys::ioreport::{
    IoReportGroups,
    decode::{Channel, RawChannel},
};

pub trait IntervalChannel: ChannelMetric {
    const CHANNEL: Channel;
    const GROUP: IoReportGroups;

    fn default_value() -> Self::Value;

    fn try_fold(
        channel: Channel,
        raw: &RawChannel,
        value: &mut Self::Value,
    ) {
        if channel == Self::CHANNEL {
            Self::fold(value, raw);
        }
    }

    fn fold(
        value: &mut Self::Value,
        raw: &RawChannel,
    );
}

pub trait IntervalSet: ChannelSet {
    const GROUPS: IoReportGroups;

    fn default_values() -> Self::Value;

    fn apply(
        channel: Channel,
        raw: &RawChannel,
        values: &mut Self::Value,
    );
}

impl IntervalSet for NilType {
    const GROUPS: IoReportGroups = IoReportGroups::empty();

    fn default_values() -> Self::Value {
        NilType
    }

    fn apply(
        _channel: Channel,
        _raw: &RawChannel,
        _values: &mut Self::Value,
    ) {
    }
}

impl<H, T> IntervalSet for ConsType<H, T>
where
    H: IntervalChannel,
    T: IntervalSet,
{
    const GROUPS: IoReportGroups = H::GROUP.union(T::GROUPS);

    fn default_values() -> Self::Value {
        Values::new(H::default_value(), T::default_values())
    }

    fn apply(
        channel: Channel,
        raw: &RawChannel,
        values: &mut Self::Value,
    ) {
        H::try_fold(channel, raw, &mut values.head);
        T::apply(channel, raw, &mut values.tail);
    }
}
