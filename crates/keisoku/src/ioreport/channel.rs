use core::ffi::c_void;

use obfstr::obfstr;
use objc2_core_foundation::{CFArray, CFDictionary, CFRetained, CFType};

use super::{IoReportFunctions, residency::ResidencyState};
use crate::cf::{cf_string_to_string, dictionary_get};

pub(super) struct ChannelSample {
    pub group: String,
    pub subgroup: String,
    pub name: String,
    pub unit: String,
    pub integer_value: i64,
    pub states: Vec<ResidencyState>,
}

pub(super) fn decode_channels(
    functions: &IoReportFunctions,
    delta: &CFDictionary,
) -> Vec<ChannelSample> {
    let Some(channels) = dictionary_get::<CFArray>(delta, obfstr!("IOReportChannels")) else {
        return Vec::new();
    };
    let channels: CFRetained<CFArray<CFType>> = unsafe { CFRetained::from_raw(CFRetained::into_raw(channels).cast()) };
    channels.iter().map(|channel| decode_channel(functions, &channel)).collect()
}

fn decode_channel(
    functions: &IoReportFunctions,
    channel: &CFType,
) -> ChannelSample {
    let item: *const c_void = core::ptr::from_ref(channel).cast();
    let states = (0..unsafe { (functions.state_get_count)(item) })
        .map(|index| ResidencyState {
            name: cf_string_to_string(unsafe { (functions.state_get_name_for_index)(item, index) }),
            residency: unsafe { (functions.state_get_residency)(item, index) },
        })
        .collect();
    ChannelSample {
        group: cf_string_to_string(unsafe { (functions.channel_get_group)(item) }),
        subgroup: cf_string_to_string(unsafe { (functions.channel_get_subgroup)(item) }),
        name: cf_string_to_string(unsafe { (functions.channel_get_channel_name)(item) }),
        unit: cf_string_to_string(unsafe { (functions.channel_get_unit_label)(item) }),
        integer_value: unsafe { (functions.simple_get_integer_value)(item, 0) },
        states,
    }
}
