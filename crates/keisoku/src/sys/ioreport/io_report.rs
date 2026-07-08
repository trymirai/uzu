use super::{
    IoReportFunctions, channel::for_each_channel, raw_energy_sample::RawEnergySample, subscription::Subscription,
};
use crate::sys::ioreport::{
    IoReportGroups,
    decode::{Channel, RawChannel},
};

pub(crate) struct IoReport {
    functions: &'static IoReportFunctions,
    subscription: Subscription,
}

impl IoReport {
    pub fn for_groups(groups: IoReportGroups) -> Option<Self> {
        let functions = IoReportFunctions::get()?;
        let subscription = Subscription::for_groups(groups, functions)?;
        Some(Self {
            functions,
            subscription,
        })
    }

    pub(crate) fn snapshot(&self) -> Option<RawEnergySample> {
        self.subscription.snapshot(self.functions).map(RawEnergySample)
    }

    pub(crate) fn for_each_channel(
        &self,
        begin: &RawEnergySample,
        end: &RawEnergySample,
        mut visit: impl FnMut(Channel, &RawChannel),
    ) {
        let Some(delta) = self.functions.create_samples_delta(&begin.0, &end.0) else {
            return;
        };
        for_each_channel(self.functions, &delta, &mut visit);
    }
}
