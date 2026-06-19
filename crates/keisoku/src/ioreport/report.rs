use core::{ffi::c_void, ptr::NonNull};
use std::time::Duration;

use obfstr::obfstr;
use objc2_core_foundation::{CFArray, CFDictionary, CFMutableDictionary, CFRetained, CFString, CFType, Type};

use super::{IoReportFunctions, SocSample};
use crate::{
    cf::{cf_dictionary_value, cf_string_to_string},
    soc::SocInfo,
};

const ANE_MAX_POWER_WATTS: f32 = 8.0;

#[derive(Clone, Copy, PartialEq)]
enum Flow {
    Read,
    Write,
    Combined,
}

#[derive(Clone, Copy)]
enum Subsystem {
    Dram,
    NeuralEngine,
}

pub struct IoReport {
    functions: &'static IoReportFunctions,
    subscription: CFRetained<CFType>,
    channels: CFRetained<CFMutableDictionary>,
}

impl IoReport {
    pub fn new() -> Option<Self> {
        let functions = IoReportFunctions::get()?;

        let mut groups: Vec<CFRetained<CFDictionary>> = Vec::with_capacity(5);
        for (group, subgroup) in [
            (obfstr!("Energy Model"), None::<&str>),
            (obfstr!("CPU Stats"), Some(obfstr!("CPU Core Performance States"))),
            (obfstr!("GPU Stats"), Some(obfstr!("GPU Performance States"))),
            (obfstr!("AMC Stats"), None),
            (obfstr!("PMP"), None),
        ] {
            if let Some(group_channels) = copy_channel_group(functions, group, subgroup) {
                groups.push(group_channels);
            }
        }

        let (first, rest) = groups.split_first()?;
        for other in rest {
            unsafe { (functions.merge_channels)(raw(first), raw(other), core::ptr::null()) };
        }
        let channels = unsafe { CFMutableDictionary::new_copy(None, first.count(), Some(&**first)) }?;

        drop(groups);
        cf_dictionary_value(&channels, obfstr!("IOReportChannels"))?;

        let mut subscribed: *mut c_void = core::ptr::null_mut();
        let subscription = unsafe {
            (functions.create_subscription)(core::ptr::null(), raw(&channels), &mut subscribed, 0, core::ptr::null())
        };
        let subscription = unsafe { CFRetained::from_raw(NonNull::new(subscription.cast_mut().cast::<CFType>())?) };

        Some(Self {
            functions,
            subscription,
            channels,
        })
    }

    pub fn sample(
        &self,
        soc: &SocInfo,
        interval: Duration,
    ) -> SocSample {
        let functions = self.functions;
        let subscription = raw(&self.subscription);
        let channels = raw(&self.channels);

        let previous = unsafe { (functions.create_samples)(subscription, channels, core::ptr::null()) };
        let Some(previous) = retained_dictionary(previous) else {
            std::thread::sleep(interval);
            return SocSample::default();
        };
        std::thread::sleep(interval);
        let next = unsafe { (functions.create_samples)(subscription, channels, core::ptr::null()) };
        let Some(next) = retained_dictionary(next) else {
            return SocSample::default();
        };
        let delta = unsafe { (functions.create_samples_delta)(raw(&previous), raw(&next), core::ptr::null()) };
        let Some(delta) = retained_dictionary(delta) else {
            return SocSample::default();
        };

        let mut result = SocSample::default();
        let mut ecpu_readings = Vec::new();
        let mut pcpu_readings = Vec::new();
        let window_milliseconds = interval.as_millis().max(1) as u64;

        let mut bandwidth = BandwidthAccumulator::default();

        if let Some(items) = cf_dictionary_value(&delta, obfstr!("IOReportChannels")) {
            let items = unsafe { &*items.cast::<CFArray>() };
            for index in 0..items.count() {
                let item = unsafe { items.value_at_index(index) };
                if item.is_null() {
                    continue;
                }
                let group = cf_string_to_string(unsafe { (functions.channel_get_group)(item) });
                let subgroup = cf_string_to_string(unsafe { (functions.channel_get_subgroup)(item) });
                let channel = cf_string_to_string(unsafe { (functions.channel_get_channel_name)(item) });

                if group == obfstr!("CPU Stats") && subgroup == obfstr!("CPU Core Performance States") {
                    if channel.starts_with(obfstr!("PCPU")) {
                        pcpu_readings.push(self.calculate_frequency(item, &soc.pcpu_frequencies));
                    } else if channel.starts_with(obfstr!("ECPU")) || channel.starts_with(obfstr!("MCPU")) {
                        ecpu_readings.push(self.calculate_frequency(item, &soc.ecpu_frequencies));
                    }
                } else if group == obfstr!("GPU Stats") && subgroup == obfstr!("GPU Performance States") {
                    if channel == obfstr!("GPUPH") && soc.gpu_frequencies.len() > 1 {
                        result.gpu_usage = self.calculate_frequency(item, &soc.gpu_frequencies[1..]);
                    }
                } else if group == obfstr!("Energy Model") {
                    let unit = cf_string_to_string(unsafe { (functions.channel_get_unit_label)(item) });
                    let watts = self.watts(item, unit.trim(), window_milliseconds);
                    if channel == obfstr!("GPU Energy") {
                        result.gpu_power += watts;
                    } else if channel.ends_with(obfstr!("CPU Energy")) {
                        result.cpu_power += watts;
                    } else if channel.starts_with(obfstr!("ANE")) {
                        result.ane_power += watts;
                    } else if channel.starts_with(obfstr!("DRAM")) {
                        result.ram_power += watts;
                    } else if channel.starts_with(obfstr!("GPU SRAM")) {
                        result.gpu_ram_power += watts;
                    }
                } else if group == obfstr!("AMC Stats") {
                    self.accumulate_amc_bandwidth(item, &channel, &mut bandwidth);
                } else if group == obfstr!("PMP") {
                    self.accumulate_pmp(item, &subgroup, &channel, &mut bandwidth, &mut result);
                }
            }
        }

        ecpu_readings.retain(|&(_, percent)| percent > 0.0);
        result.ecpu_usage = average_cluster_frequency(&ecpu_readings, &soc.ecpu_frequencies);
        result.pcpu_usage = average_cluster_frequency(&pcpu_readings, &soc.pcpu_frequencies);
        let efficiency_cores = soc.ecpu_cores as f32;
        let performance_cores = soc.pcpu_cores as f32;
        result.cpu_usage_percent = divide_or_zero(
            result.ecpu_usage.1 * efficiency_cores + result.pcpu_usage.1 * performance_cores,
            efficiency_cores + performance_cores,
        );
        result.total_power = result.cpu_power + result.gpu_power + result.ane_power;

        bandwidth.finish(window_milliseconds, &mut result);
        if result.ane_active_percent == 0.0 && result.ane_power > 0.0 {
            result.ane_active_percent = (result.ane_power / ANE_MAX_POWER_WATTS * 100.0).min(100.0);
        }
        result
    }

    fn accumulate_amc_bandwidth(
        &self,
        item: *const c_void,
        channel: &str,
        bandwidth: &mut BandwidthAccumulator,
    ) {
        let aggregate = strip_die_prefix(channel);
        let target = if aggregate == obfstr!("DCS RD") {
            (Subsystem::Dram, Flow::Read)
        } else if aggregate == obfstr!("DCS WR") {
            (Subsystem::Dram, Flow::Write)
        } else if aggregate == obfstr!("ANE DCS RD") {
            (Subsystem::NeuralEngine, Flow::Read)
        } else if aggregate == obfstr!("ANE DCS WR") {
            (Subsystem::NeuralEngine, Flow::Write)
        } else {
            return;
        };
        let bytes = unsafe { (self.functions.simple_get_integer_value)(item, 0) } as f64;
        if bytes <= 0.0 {
            return;
        }
        match target.0 {
            Subsystem::Dram => bandwidth.add_dram_bytes(target.1, bytes),
            Subsystem::NeuralEngine => bandwidth.add_ane_bytes(target.1, bytes),
        }
    }

    fn accumulate_pmp(
        &self,
        item: *const c_void,
        subgroup: &str,
        channel: &str,
        bandwidth: &mut BandwidthAccumulator,
        result: &mut SocSample,
    ) {
        if subgroup.contains(obfstr!("Floor")) && (channel == obfstr!("ANE-AF-BW") || channel == obfstr!("ANE-DCS-BW"))
        {
            result.ane_active_percent = result.ane_active_percent.max(self.residency_active_percent(item));
        } else if subgroup == obfstr!("AF BW") {
            if channel == obfstr!("ANE0 RD") {
                bandwidth.ane_read_histogram_gbps =
                    bandwidth.ane_read_histogram_gbps.max(self.residency_weighted_gbps(item));
            } else if channel == obfstr!("ANE0 WR") {
                bandwidth.ane_write_histogram_gbps =
                    bandwidth.ane_write_histogram_gbps.max(self.residency_weighted_gbps(item));
            }
        } else if subgroup == obfstr!("DRAM BW") {
            let gbps = self.residency_weighted_gbps(item);
            match flow(channel) {
                Flow::Read => bandwidth.dram_read_histogram_gbps = bandwidth.dram_read_histogram_gbps.max(gbps),
                Flow::Write => bandwidth.dram_write_histogram_gbps = bandwidth.dram_write_histogram_gbps.max(gbps),
                Flow::Combined => {},
            }
        }
    }

    fn residencies(
        &self,
        item: *const c_void,
    ) -> Vec<(String, i64)> {
        let state_count = unsafe { (self.functions.state_get_count)(item) };
        (0..state_count)
            .map(|index| {
                let name = cf_string_to_string(unsafe { (self.functions.state_get_name_for_index)(item, index) });
                let residency = unsafe { (self.functions.state_get_residency)(item, index) };
                (name, residency)
            })
            .collect()
    }

    fn residency_active_percent(
        &self,
        item: *const c_void,
    ) -> f32 {
        let states = self.residencies(item);
        let total: f64 = states.iter().map(|(_, residency)| *residency as f64).sum();
        if total <= 0.0 {
            return 0.0;
        }
        let active: f64 =
            states.iter().filter(|(name, _)| !is_idle_state(name)).map(|(_, residency)| *residency as f64).sum();
        (active / total * 100.0) as f32
    }

    fn residency_weighted_gbps(
        &self,
        item: *const c_void,
    ) -> f32 {
        let states = self.residencies(item);
        let mut weighted = 0f64;
        let mut total = 0f64;
        for (name, residency) in &states {
            weighted += parse_leading_number(name) * (*residency as f64);
            total += *residency as f64;
        }
        if total <= 0.0 {
            0.0
        } else {
            (weighted / total) as f32
        }
    }

    fn watts(
        &self,
        item: *const c_void,
        unit: &str,
        window_milliseconds: u64,
    ) -> f32 {
        let energy = unsafe { (self.functions.simple_get_integer_value)(item, 0) } as f32;
        let power = energy / (window_milliseconds as f32 / 1000.0);
        if unit == obfstr!("mJ") {
            power / 1e3
        } else if unit == obfstr!("uJ") {
            power / 1e6
        } else if unit == obfstr!("nJ") {
            power / 1e9
        } else {
            0.0
        }
    }

    fn calculate_frequency(
        &self,
        item: *const c_void,
        frequencies: &[u32],
    ) -> (u32, f32) {
        let states = self.residencies(item);
        if states.len() <= frequencies.len() || frequencies.is_empty() {
            return (0, 0.0);
        }

        let Some(offset) = states.iter().position(|(name, _)| !is_idle_state(name)) else {
            return (0, 0.0);
        };
        let active: f64 = states.iter().skip(offset).map(|(_, residency)| *residency as f64).sum();
        let total: f64 = states.iter().map(|(_, residency)| *residency as f64).sum();

        let mut average_frequency = 0f64;
        for (index, &frequency) in frequencies.iter().enumerate() {
            let percent = divide_or_zero(states[index + offset].1 as f64, active);
            average_frequency += percent * frequency as f64;
        }
        let usage_ratio = divide_or_zero(active, total);
        let minimum_frequency = *frequencies.first().unwrap() as f64;
        let maximum_frequency = *frequencies.last().unwrap() as f64;
        let fraction_of_max = (average_frequency.max(minimum_frequency) * usage_ratio) / maximum_frequency;
        (average_frequency as u32, fraction_of_max as f32)
    }
}

fn raw<T: Type>(value: &CFRetained<T>) -> *mut c_void {
    CFRetained::as_ptr(value).as_ptr().cast()
}

fn retained_dictionary(value: *const c_void) -> Option<CFRetained<CFDictionary>> {
    NonNull::new(value.cast_mut().cast::<CFDictionary>()).map(|pointer| unsafe { CFRetained::from_raw(pointer) })
}

#[derive(Default)]
struct BandwidthAccumulator {
    dram_read_bytes: f64,
    dram_write_bytes: f64,
    dram_combined_bytes: f64,
    ane_read_bytes: f64,
    ane_write_bytes: f64,
    ane_combined_bytes: f64,
    dram_read_histogram_gbps: f32,
    dram_write_histogram_gbps: f32,
    ane_read_histogram_gbps: f32,
    ane_write_histogram_gbps: f32,
}

impl BandwidthAccumulator {
    fn add_dram_bytes(
        &mut self,
        flow: Flow,
        bytes: f64,
    ) {
        match flow {
            Flow::Read => self.dram_read_bytes += bytes,
            Flow::Write => self.dram_write_bytes += bytes,
            Flow::Combined => self.dram_combined_bytes += bytes,
        }
    }

    fn add_ane_bytes(
        &mut self,
        flow: Flow,
        bytes: f64,
    ) {
        match flow {
            Flow::Read => self.ane_read_bytes += bytes,
            Flow::Write => self.ane_write_bytes += bytes,
            Flow::Combined => self.ane_combined_bytes += bytes,
        }
    }

    fn finish(
        mut self,
        window_milliseconds: u64,
        result: &mut SocSample,
    ) {
        let window_seconds = (window_milliseconds as f64 / 1000.0).max(0.001);
        let to_gbps = |bytes: f64| (bytes / window_seconds / 1e9) as f32;

        if self.dram_read_bytes == 0.0 && self.dram_write_bytes == 0.0 && self.dram_combined_bytes > 0.0 {
            self.dram_read_bytes = self.dram_combined_bytes / 2.0;
            self.dram_write_bytes = self.dram_combined_bytes / 2.0;
        }
        if self.ane_read_bytes == 0.0 && self.ane_write_bytes == 0.0 && self.ane_combined_bytes > 0.0 {
            self.ane_read_bytes = self.ane_combined_bytes / 2.0;
            self.ane_write_bytes = self.ane_combined_bytes / 2.0;
        }

        result.dram_read_gbps = if self.dram_read_bytes > 0.0 {
            to_gbps(self.dram_read_bytes)
        } else {
            self.dram_read_histogram_gbps
        };
        result.dram_write_gbps = if self.dram_write_bytes > 0.0 {
            to_gbps(self.dram_write_bytes)
        } else {
            self.dram_write_histogram_gbps
        };
        result.ane_read_gbps = if self.ane_read_bytes > 0.0 {
            to_gbps(self.ane_read_bytes)
        } else {
            self.ane_read_histogram_gbps
        };
        result.ane_write_gbps = if self.ane_write_bytes > 0.0 {
            to_gbps(self.ane_write_bytes)
        } else {
            self.ane_write_histogram_gbps
        };
    }
}

fn copy_channel_group(
    functions: &IoReportFunctions,
    group: &str,
    subgroup: Option<&str>,
) -> Option<CFRetained<CFDictionary>> {
    let group_name = CFString::from_str(group);
    let subgroup_name = subgroup.map(CFString::from_str);
    let subgroup_pointer = match &subgroup_name {
        Some(name) => raw(name),
        None => core::ptr::null_mut(),
    };
    let channels = unsafe { (functions.copy_channels_in_group)(raw(&group_name), subgroup_pointer, 0, 0, 0) };
    retained_dictionary(channels)
}

fn strip_die_prefix(channel: &str) -> &str {
    let Some(rest) = channel.strip_prefix(obfstr!("DIE")) else {
        return channel;
    };
    let rest = rest.trim_start_matches(|character: char| character.is_ascii_digit());
    rest.strip_prefix(' ').unwrap_or(channel)
}

fn flow(channel: &str) -> Flow {
    if channel.contains(obfstr!("RD+WR")) || channel.ends_with(obfstr!("RW")) {
        Flow::Combined
    } else if channel.contains(obfstr!("WR")) {
        Flow::Write
    } else if channel.contains(obfstr!("RD")) {
        Flow::Read
    } else {
        Flow::Combined
    }
}

fn is_idle_state(name: &str) -> bool {
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

fn average_cluster_frequency(
    readings: &[(u32, f32)],
    frequencies: &[u32],
) -> (u32, f32) {
    if readings.is_empty() || frequencies.is_empty() {
        return (0, 0.0);
    }
    let average_frequency =
        divide_or_zero(readings.iter().map(|reading| reading.0 as f32).sum::<f32>(), readings.len() as f32);
    let average_percent = divide_or_zero(readings.iter().map(|reading| reading.1).sum::<f32>(), readings.len() as f32);
    let minimum_frequency = *frequencies.first().unwrap() as f32;
    (average_frequency.max(minimum_frequency) as u32, average_percent)
}

fn divide_or_zero<T: core::ops::Div<Output = T> + Default + PartialEq>(
    numerator: T,
    denominator: T,
) -> T {
    let zero = T::default();
    if denominator == zero {
        zero
    } else {
        numerator / denominator
    }
}
