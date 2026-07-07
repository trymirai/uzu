use obfstr::obfstr;
use objc2_core_foundation::{CFDictionary, CFMutableDictionary, CFRetained, CFString, CFType};

use super::IoReportFunctions;
use crate::metric::IoReportGroups;

pub(super) struct Channels(CFRetained<CFMutableDictionary>);

impl Channels {
    fn from_groups(
        groups: IoReportGroups,
        functions: &IoReportFunctions,
    ) -> Option<Self> {
        let mut collected: Vec<CFRetained<CFMutableDictionary>> = Vec::with_capacity(5);
        for (flag, group, subgroup) in [
            (IoReportGroups::ENERGY_MODEL, obfstr!("Energy Model"), None::<&str>),
            (IoReportGroups::CPU_STATS, obfstr!("CPU Stats"), Some(obfstr!("CPU Core Performance States"))),
            (IoReportGroups::GPU_STATS, obfstr!("GPU Stats"), Some(obfstr!("GPU Performance States"))),
            (IoReportGroups::AMC_STATS, obfstr!("AMC Stats"), None),
            (IoReportGroups::PMP, obfstr!("PMP"), None),
        ] {
            if !groups.contains(flag) {
                continue;
            }
            let group_name = CFString::from_str(group);
            let subgroup_name = subgroup.map(CFString::from_str);
            if let Some(group_channels) = functions.copy_channels_in_group(&group_name, subgroup_name.as_deref()) {
                collected.push(group_channels);
            }
        }
        Self::merged(&collected, functions)
    }

    fn merged(
        groups: &[CFRetained<CFMutableDictionary>],
        functions: &IoReportFunctions,
    ) -> Option<Self> {
        let (first, rest) = groups.split_first()?;
        for other in rest {
            functions.merge_channels(first, other);
        }
        let channels = unsafe { CFMutableDictionary::new_copy(None, first.count(), Some(&**first)) }?;
        Some(Self(channels))
    }
}

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
