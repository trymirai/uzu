use core::ffi::c_void;

use obfstr::obfstr;
use objc2_core_foundation::{CFArray, CFDictionary, CFRetained, CFType};

use super::IoReportFunctions;
use crate::cf::{cf_string_to_string, dictionary_get};

pub(super) struct ResidencyState {
    pub name: String,
    pub residency: i64,
}

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

pub(super) fn residency_active_percent(states: &[ResidencyState]) -> f32 {
    let total: f64 = states.iter().map(|state| state.residency as f64).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let active: f64 =
        states.iter().filter(|state| !is_idle_state(&state.name)).map(|state| state.residency as f64).sum();
    (active / total * 100.0) as f32
}

pub(super) fn residency_weighted_gbps(states: &[ResidencyState]) -> f32 {
    let mut weighted = 0f64;
    let mut total = 0f64;
    for state in states {
        weighted += parse_leading_number(&state.name) * (state.residency as f64);
        total += state.residency as f64;
    }
    if total <= 0.0 {
        0.0
    } else {
        (weighted / total) as f32
    }
}

pub(super) fn is_idle_state(name: &str) -> bool {
    name == obfstr!("OFF")
        || name == obfstr!("IDLE")
        || name == obfstr!("DOWN")
        || name == obfstr!("SLEEP")
        || name == obfstr!("VMIN")
        || name == obfstr!("F1")
        || name == obfstr!("0%")
}

fn parse_leading_number(name: &str) -> f64 {
    let digits: String =
        name.trim().chars().take_while(|character| character.is_ascii_digit() || *character == '.').collect();
    digits.parse().unwrap_or(0.0)
}
