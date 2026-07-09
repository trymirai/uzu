use super::{
    IoReportFunctions, channel::for_each_channel, raw_ioreport_sample::RawIOReportSample, subscription::Subscription,
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

    pub(crate) fn snapshot(&self) -> Option<RawIOReportSample> {
        self.subscription.snapshot(self.functions).map(RawIOReportSample)
    }

    pub(crate) fn for_each_channel(
        &self,
        begin: &RawIOReportSample,
        end: &RawIOReportSample,
        mut visit: impl FnMut(Channel, &RawChannel),
    ) {
        let Some(delta) = self.functions.create_samples_delta(&begin.0, &end.0) else {
            return;
        };
        for_each_channel(self.functions, &delta, &mut visit);
    }
}
