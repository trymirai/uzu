use core::marker::PhantomData;

use super::super::{channel_set::IntervalChannel, typelist::ChannelMetric};
use crate::{
    sys::ioreport::{
        IoReportGroups,
        decode::{Channel, RawChannel, residency_weighted_gbps},
        kinds::FlowKind,
    },
    units::GigabytesPerSecond,
};

pub struct DramHistogram<F>(PhantomData<F>);

impl<F: FlowKind> ChannelMetric for DramHistogram<F> {
    type Value = GigabytesPerSecond;
    const TYPE_BIT: u128 = F::TYPE_BIT_HIST;
}

impl<F: FlowKind> IntervalChannel for DramHistogram<F> {
    const CHANNEL: Channel = Channel::DramHistogram(F::FLOW);
    const GROUP: IoReportGroups = IoReportGroups::PMP;

    fn default_value() -> Self::Value {
        GigabytesPerSecond(0.0)
    }

    fn fold(
        value: &mut Self::Value,
        raw: &RawChannel,
    ) {
        *value = GigabytesPerSecond(value.value().max(residency_weighted_gbps(&raw.states)));
    }
}
