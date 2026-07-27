use core::marker::PhantomData;

use super::super::{channel_set::IntervalChannel, typelist::ChannelMetric};
use crate::{
    sys::ioreport::{
        IoReportGroups,
        decode::{Channel, RawChannel},
        kinds::FlowKind,
    },
    units::Bytes,
};

pub struct DramBytes<F>(PhantomData<F>);

impl<F: FlowKind> ChannelMetric for DramBytes<F> {
    type Value = Bytes;
    const TYPE_BIT: u128 = F::TYPE_BIT_BYTES;
}

impl<F: FlowKind> IntervalChannel for DramBytes<F> {
    const CHANNEL: Channel = Channel::DramBytes(F::FLOW);
    const GROUP: IoReportGroups = IoReportGroups::AMC_STATS.union(IoReportGroups::PMP);

    fn default_value() -> Self::Value {
        Bytes(0)
    }

    fn fold(
        value: &mut Self::Value,
        raw: &RawChannel,
    ) {
        let bytes = raw.integer_value;
        if bytes > 0 {
            *value = Bytes(value.value() + bytes as u64);
        }
    }
}
