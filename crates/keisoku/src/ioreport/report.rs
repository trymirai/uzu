use objc2_core_foundation::{CFDictionary, CFRetained};

use super::{IoReportFunctions, channel::decode_channels, subscription::Subscription};
use crate::{decode::ChannelSample, metric::IoReportGroups};

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

    pub(crate) fn decode(
        &self,
        begin: &RawEnergySample,
        end: &RawEnergySample,
    ) -> Box<[ChannelSample]> {
        match self.functions.create_samples_delta(&begin.0, &end.0) {
            Some(delta) => decode_channels(self.functions, &delta),
            None => Box::default(),
        }
    }
}
