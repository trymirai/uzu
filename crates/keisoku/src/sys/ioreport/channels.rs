use obfstr::obfstr;
use objc2_core_foundation::{CFMutableDictionary, CFRetained, CFString};

use super::IoReportFunctions;
use crate::sys::ioreport::IoReportGroups;

pub(super) struct Channels(pub(super) CFRetained<CFMutableDictionary>);

impl Channels {
    pub(super) fn from_groups(
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
