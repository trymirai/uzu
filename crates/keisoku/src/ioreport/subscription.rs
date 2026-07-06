use obfstr::obfstr;
use objc2_core_foundation::{CFDictionary, CFMutableDictionary, CFRetained, CFString, CFType};

use super::IoReportFunctions;

pub(super) struct Channels(CFRetained<CFMutableDictionary>);

impl Channels {
    fn all(functions: &IoReportFunctions) -> Option<Self> {
        let mut groups: Vec<CFRetained<CFMutableDictionary>> = Vec::with_capacity(5);
        for (group, subgroup) in [
            (obfstr!("Energy Model"), None::<&str>),
            (obfstr!("CPU Stats"), Some(obfstr!("CPU Core Performance States"))),
            (obfstr!("GPU Stats"), Some(obfstr!("GPU Performance States"))),
            (obfstr!("AMC Stats"), None),
            (obfstr!("PMP"), None),
        ] {
            let group_name = CFString::from_str(group);
            let subgroup_name = subgroup.map(CFString::from_str);
            if let Some(group_channels) = functions.copy_channels_in_group(&group_name, subgroup_name.as_deref()) {
                groups.push(group_channels);
            }
        }
        Self::merged(&groups, functions)
    }

    fn energy_model(functions: &IoReportFunctions) -> Option<Self> {
        let group_name = CFString::from_str(obfstr!("Energy Model"));
        let group_channels = functions.copy_channels_in_group(&group_name, None)?;
        Self::merged(&[group_channels], functions)
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
    pub(super) fn new(functions: &IoReportFunctions) -> Option<Self> {
        Self::over(Channels::all(functions)?, functions)
    }

    pub(super) fn energy_model(functions: &IoReportFunctions) -> Option<Self> {
        Self::over(Channels::energy_model(functions)?, functions)
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
