use core::ffi::c_void;

use obfstr::obfstr;
use objc2_core_foundation::{CFArray, CFDictionary, CFRetained, CFType};

use super::IoReportFunctions;
use crate::sys::{
    ioreport::decode::{Channel, GroupId, RawChannel, ResidencyState},
    registry::{cf_string_to_string, dictionary_get},
};

pub(super) fn for_each_channel(
    functions: &IoReportFunctions,
    delta: &CFDictionary,
    visit: &mut impl FnMut(Channel, &RawChannel),
) {
    let Some(channels) = dictionary_get::<CFArray>(delta, obfstr!("IOReportChannels")) else {
        return;
    };
    let channels: CFRetained<CFArray<CFType>> = unsafe { CFRetained::from_raw(CFRetained::into_raw(channels).cast()) };
    for channel in channels.iter() {
        if let Some((classified, decoded)) = decode_channel(functions, &channel) {
            visit(classified, &decoded);
        }
    }
}

fn decode_channel(
    functions: &IoReportFunctions,
    channel: &CFType,
) -> Option<(Channel, RawChannel)> {
    let item: *const c_void = core::ptr::from_ref(channel).cast();
    let group = GroupId::classify(&cf_string_to_string(unsafe { (functions.channel_get_group)(item) }));
    if group == GroupId::Other {
        return None;
    }

    let mut decoded = RawChannel {
        group,
        name: cf_string_to_string(unsafe { (functions.channel_get_channel_name)(item) }),
        ..Default::default()
    };
    if matches!(group, GroupId::CpuStats | GroupId::GpuStats | GroupId::Pmp) {
        decoded.subgroup = cf_string_to_string(unsafe { (functions.channel_get_subgroup)(item) });
    }

    let classified = Channel::classify(&decoded)?;

    match classified {
        Channel::EnergyRail(_) => {
            decoded.unit = unsafe { (functions.channel_get_unit)(item) };
            decoded.integer_value = unsafe { (functions.simple_get_integer_value)(item, 0) };
        },
        Channel::DramBytes(_) => {
            decoded.integer_value = unsafe { (functions.simple_get_integer_value)(item, 0) };
        },
        Channel::CpuCluster(_) | Channel::GpuState | Channel::AneBandwidth | Channel::DramHistogram(_) => {
            decoded.states = decode_states(functions, item);
        },
    }

    Some((classified, decoded))
}

fn decode_states(
    functions: &IoReportFunctions,
    item: *const c_void,
) -> Vec<ResidencyState> {
    (0..unsafe { (functions.state_get_count)(item) })
        .map(|index| ResidencyState {
            name: cf_string_to_string(unsafe { (functions.state_get_name_for_index)(item, index) }),
            residency: unsafe { (functions.state_get_residency)(item, index) },
        })
        .collect()
}
