use objc2_core_foundation::{CFDictionary, CFRetained};

use super::{IoReportFunctions, channel::for_each_channel, subscription::Subscription};
use crate::{decode::RawChannel, metric::IoReportGroups};

pub struct IoReport {
    functions: &'static IoReportFunctions,
    subscription: Subscription,
}

pub(crate) struct RawEnergySample(CFRetained<CFDictionary>);

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

    /// Diff the begin/end snapshots and hand each decoded channel to `visit` in a
    /// single pass — no intermediate collection.
    pub(crate) fn for_each_channel(
        &self,
        begin: &RawEnergySample,
        end: &RawEnergySample,
        mut visit: impl FnMut(&RawChannel),
    ) {
        let Some(delta) = self.functions.create_samples_delta(&begin.0, &end.0) else {
            return;
        };
        for_each_channel(self.functions, &delta, &mut visit);
    }
}
