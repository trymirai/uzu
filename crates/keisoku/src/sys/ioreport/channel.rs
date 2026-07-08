use core::ffi::c_void;

use obfstr::obfstr;
use objc2_core_foundation::{CFArray, CFDictionary, CFRetained, CFType};

use super::IoReportFunctions;
use crate::sys::{
    ioreport::decode::{GroupId, RawChannel, ResidencyState},
    registry::{cf_string_to_string, dictionary_get},
};

pub(super) fn for_each_channel(
    functions: &IoReportFunctions,
    delta: &CFDictionary,
    wants: impl Fn(&RawChannel) -> bool,
    visit: &mut impl FnMut(&RawChannel),
) {
    let Some(channels) = dictionary_get::<CFArray>(delta, obfstr!("IOReportChannels")) else {
        return;
    };
    let channels: CFRetained<CFArray<CFType>> = unsafe { CFRetained::from_raw(CFRetained::into_raw(channels).cast()) };
    for channel in channels.iter() {
        let Some(decoded) = decode_channel(functions, &channel, &wants) else {
            continue;
        };
        visit(&decoded);
    }
}

/// Decode only the fields a channel needs. State residency tables are read only
/// after `wants` accepts the channel header (group, subgroup, name).
fn decode_channel(
    functions: &IoReportFunctions,
    channel: &CFType,
    wants: &impl Fn(&RawChannel) -> bool,
) -> Option<RawChannel> {
    let item: *const c_void = core::ptr::from_ref(channel).cast();
    let group = GroupId::classify(&cf_string_to_string(unsafe { (functions.channel_get_group)(item) }));
    if group == GroupId::Other {
        return None;
    }

    let mut decoded = RawChannel {
        group,
        ..Default::default()
    };

    match group {
        GroupId::EnergyModel => {
            decoded.name = cf_string_to_string(unsafe { (functions.channel_get_channel_name)(item) });
            decoded.unit = cf_string_to_string(unsafe { (functions.channel_get_unit_label)(item) });
            if !wants(&decoded) {
                return None;
            }
            decoded.integer_value = unsafe { (functions.simple_get_integer_value)(item, 0) };
        },
        GroupId::AmcStats => {
            decoded.name = cf_string_to_string(unsafe { (functions.channel_get_channel_name)(item) });
            if !wants(&decoded) {
                return None;
            }
            decoded.integer_value = unsafe { (functions.simple_get_integer_value)(item, 0) };
        },
        GroupId::CpuStats | GroupId::GpuStats | GroupId::Pmp => {
            decoded.subgroup = cf_string_to_string(unsafe { (functions.channel_get_subgroup)(item) });
            decoded.name = cf_string_to_string(unsafe { (functions.channel_get_channel_name)(item) });
            if !wants(&decoded) {
                return None;
            }
            decoded.states = decode_states(functions, item);
        },
        GroupId::Other => return None,
    }

    Some(decoded)
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
