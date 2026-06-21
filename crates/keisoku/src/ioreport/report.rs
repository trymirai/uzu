use std::time::{Duration, Instant};

use obfstr::obfstr;
use objc2_core_foundation::{CFDictionary, CFMutableDictionary, CFRetained, CFString, CFType};

use super::{
    IoReportFunctions, SocSample,
    bandwidth::{BandwidthAccumulator, accumulate_amc_bandwidth, accumulate_pmp},
    channel::decode_channels,
    frequency::{average_cluster_frequency, calculate_frequency, divide_or_zero},
};
use crate::soc::SocInfo;

const ANE_MAX_POWER_WATTS: f32 = 8.0;

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
            let group_name = CFString::from_str(group);
            let subgroup_name = subgroup.map(CFString::from_str);
            if let Some(group_channels) = functions.copy_channels_in_group(&group_name, subgroup_name.as_deref()) {
                groups.push(group_channels);
            }
        }

        let (first, rest) = groups.split_first()?;
        for other in rest {
            functions.merge_channels(first, other);
        }
        let channels = unsafe { CFMutableDictionary::new_copy(None, first.count(), Some(&**first)) }?;

        drop(groups);

        let (subscription, subscribed) = functions.create_subscription(&channels)?;
        drop(subscribed);

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

        let Some(previous) = functions.create_samples(&self.subscription, &self.channels) else {
            std::thread::sleep(interval);
            return SocSample::default();
        };
        let started = Instant::now();
        std::thread::sleep(interval);
        let next = functions.create_samples(&self.subscription, &self.channels);
        let elapsed = started.elapsed();
        let Some(next) = next else {
            return SocSample::default();
        };
        let Some(delta) = functions.create_samples_delta(&previous, &next) else {
            return SocSample::default();
        };

        let mut result = SocSample::default();
        let mut ecpu_readings = Vec::new();
        let mut pcpu_readings = Vec::new();
        let window_milliseconds = elapsed.as_millis().max(1) as u64;
        let mut bandwidth = BandwidthAccumulator::default();

        for channel in decode_channels(functions, &delta) {
            if channel.group == obfstr!("CPU Stats") && channel.subgroup == obfstr!("CPU Core Performance States") {
                if channel.name.starts_with(obfstr!("PCPU")) {
                    pcpu_readings.push(calculate_frequency(&channel.states, &soc.pcpu_frequencies));
                } else if channel.name.starts_with(obfstr!("ECPU")) || channel.name.starts_with(obfstr!("MCPU")) {
                    ecpu_readings.push(calculate_frequency(&channel.states, &soc.ecpu_frequencies));
                }
            } else if channel.group == obfstr!("GPU Stats") && channel.subgroup == obfstr!("GPU Performance States") {
                if channel.name == obfstr!("GPUPH") && soc.gpu_frequencies.len() > 1 {
                    result.gpu_usage = calculate_frequency(&channel.states, &soc.gpu_frequencies[1..]);
                }
            } else if channel.group == obfstr!("Energy Model") {
                let watts = watts(channel.integer_value, channel.unit.trim(), window_milliseconds);
                if channel.name == obfstr!("GPU Energy") {
                    result.gpu_power += watts;
                } else if channel.name.ends_with(obfstr!("CPU Energy")) {
                    result.cpu_power += watts;
                } else if channel.name.starts_with(obfstr!("ANE")) {
                    result.ane_power += watts;
                } else if channel.name.starts_with(obfstr!("DRAM")) {
                    result.ram_power += watts;
                } else if channel.name.starts_with(obfstr!("GPU SRAM")) {
                    result.gpu_ram_power += watts;
                }
            } else if channel.group == obfstr!("AMC Stats") {
                accumulate_amc_bandwidth(channel.integer_value, &channel.name, &mut bandwidth);
            } else if channel.group == obfstr!("PMP") {
                accumulate_pmp(&channel.states, &channel.subgroup, &channel.name, &mut bandwidth, &mut result);
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
}

fn watts(
    energy: i64,
    unit: &str,
    window_milliseconds: u64,
) -> f32 {
    let power = energy as f32 / (window_milliseconds as f32 / 1000.0);
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
