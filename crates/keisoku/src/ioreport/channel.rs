use core::ffi::c_void;

use obfstr::obfstr;
use objc2_core_foundation::{CFArray, CFDictionary, CFRetained, CFType};

use super::IoReportFunctions;
use crate::{
    cf::{cf_string_to_string, dictionary_get},
    decode::{GroupId, RawChannel, ResidencyState},
};

pub(super) fn for_each_channel(
    functions: &IoReportFunctions,
    delta: &CFDictionary,
    visit: &mut impl FnMut(&RawChannel),
) {
    let Some(channels) = dictionary_get::<CFArray>(delta, obfstr!("IOReportChannels")) else {
        return;
    };
    let channels: CFRetained<CFArray<CFType>> = unsafe { CFRetained::from_raw(CFRetained::into_raw(channels).cast()) };
    for channel in channels.iter() {
        let decoded = decode_channel(functions, &channel);
        visit(&decoded);
    }
}

/// Decode only the fields a channel's group actually uses: energy/byte-counter
/// channels get name/unit/value (no residency states), state-based channels get
/// subgroup/name/states, and everything else is skipped.
fn decode_channel(
    functions: &IoReportFunctions,
    channel: &CFType,
) -> RawChannel {
    let item: *const c_void = core::ptr::from_ref(channel).cast();
    let group = GroupId::classify(&cf_string_to_string(unsafe { (functions.channel_get_group)(item) }));
    let mut decoded = RawChannel {
        group,
        ..Default::default()
    };
    match group {
        GroupId::EnergyModel => {
            decoded.name = cf_string_to_string(unsafe { (functions.channel_get_channel_name)(item) });
            decoded.unit = cf_string_to_string(unsafe { (functions.channel_get_unit_label)(item) });
            decoded.integer_value = unsafe { (functions.simple_get_integer_value)(item, 0) };
        },
        GroupId::AmcStats => {
            decoded.name = cf_string_to_string(unsafe { (functions.channel_get_channel_name)(item) });
            decoded.integer_value = unsafe { (functions.simple_get_integer_value)(item, 0) };
        },
        GroupId::CpuStats | GroupId::GpuStats | GroupId::Pmp => {
            decoded.subgroup = cf_string_to_string(unsafe { (functions.channel_get_subgroup)(item) });
            decoded.name = cf_string_to_string(unsafe { (functions.channel_get_channel_name)(item) });
            decoded.states = decode_states(functions, item);
        },
        GroupId::Other => {},
    }
    decoded
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
