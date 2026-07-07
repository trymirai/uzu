use objc2_core_foundation::{CFDictionary, CFRetained, CFType};

use super::{IoReportFunctions, channels::Channels};
use crate::metric::IoReportGroups;

pub(super) struct Subscription {
    handle: CFRetained<CFType>,
    channels: Channels,
}

impl Subscription {
    pub(super) fn for_groups(
        groups: IoReportGroups,
        functions: &IoReportFunctions,
    ) -> Option<Self> {
        Self::over(Channels::from_groups(groups, functions)?, functions)
    }

    fn over(
        channels: Channels,
        functions: &IoReportFunctions,
    ) -> Option<Self> {
        let (handle, subscribed) = functions.create_subscription(&channels.0)?;
        drop(subscribed);
        Some(Self {
            handle,
            channels,
        })
    }

    pub(super) fn snapshot(
        &self,
        functions: &IoReportFunctions,
    ) -> Option<CFRetained<CFDictionary>> {
        functions.create_samples(&self.handle, &self.channels.0)
    }
}
