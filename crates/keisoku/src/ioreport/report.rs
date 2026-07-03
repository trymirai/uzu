use std::time::{Duration, Instant};

use obfstr::obfstr;
use objc2_core_foundation::{CFDictionary, CFRetained};

use super::{
    IoReportFunctions, SocSample,
    bandwidth::{BandwidthAccumulator, accumulate_amc_bandwidth, accumulate_pmp},
    channel::decode_channels,
    frequency::{average_cluster_frequency, calculate_frequency, divide_or_zero},
    subscription::Subscription,
};
use crate::{
    EnergyMetrics, PowerMetrics,
    soc::SocInfo,
    units::{Joules, Watts},
};

const ANE_MAX_POWER_WATTS: f32 = 8.0;

pub struct IoReport {
    functions: &'static IoReportFunctions,
    subscription: Subscription,
}

pub(crate) struct RawEnergySample(CFRetained<CFDictionary>);

#[derive(Default)]
pub(crate) struct EnergyTotals {
    cpu: f32,
    gpu: f32,
    gpu_sram: f32,
    ane: f32,
    ram: f32,
}

impl EnergyTotals {
    pub(crate) fn total(&self) -> f32 {
        self.cpu + self.gpu + self.gpu_sram + self.ane + self.ram
    }

    pub(crate) fn energy_metrics(
        &self,
        package: Joules,
    ) -> EnergyMetrics {
        EnergyMetrics {
            cpu: Joules(self.cpu),
            gpu: Joules(self.gpu),
            gpu_sram: Joules(self.gpu_sram),
            ane: Joules(self.ane),
            ram: Joules(self.ram),
            package,
        }
    }

    pub(crate) fn power_metrics(
        &self,
        elapsed: Duration,
        package: Watts,
    ) -> PowerMetrics {
        let elapsed_secs = elapsed.as_secs_f32().max(0.001);
        PowerMetrics {
            cpu: Watts(self.cpu / elapsed_secs),
            gpu: Watts(self.gpu / elapsed_secs),
            gpu_sram: Watts(self.gpu_sram / elapsed_secs),
            ane: Watts(self.ane / elapsed_secs),
            ram: Watts(self.ram / elapsed_secs),
            total: Watts(self.total() / elapsed_secs),
            package,
        }
    }
}

impl IoReport {
    pub fn new() -> Option<Self> {
        let functions = IoReportFunctions::get()?;
        let subscription = Subscription::new(functions)?;
        Some(Self {
            functions,
            subscription,
        })
    }

    pub(crate) fn snapshot(&self) -> Option<RawEnergySample> {
        self.subscription.snapshot(self.functions).map(RawEnergySample)
    }

    pub(crate) fn energy_delta(
        &self,
        before: &RawEnergySample,
        after: &RawEnergySample,
    ) -> Option<EnergyTotals> {
        let delta = self.functions.create_samples_delta(&before.0, &after.0)?;
        Some(energy_totals(self.functions, &delta))
    }

    pub fn sample(
        &self,
        soc: &SocInfo,
        interval: Duration,
    ) -> SocSample {
        let functions = self.functions;

        let Some(previous) = self.subscription.snapshot(functions) else {
            std::thread::sleep(interval);
            return SocSample::default();
        };
        let started = Instant::now();
        std::thread::sleep(interval);
        let next = self.subscription.snapshot(functions);
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
        let mut energy = EnergyTotals::default();

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
                accumulate_energy(&mut energy, &channel.name, joules(channel.integer_value, channel.unit.trim()));
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
        let window = Duration::from_millis(window_milliseconds);
        let package = Watts(energy.total() / window.as_secs_f32().max(0.001));
        let power = energy.power_metrics(window, package);
        result.cpu_power = power.cpu.value();
        result.gpu_power = power.gpu.value();
        result.gpu_ram_power = power.gpu_sram.value();
        result.ane_power = power.ane.value();
        result.ram_power = power.ram.value();
        result.total_power = power.total.value();

        bandwidth.finish(window_milliseconds, &mut result);
        if result.ane_active_percent == 0.0 && result.ane_power > 0.0 {
            result.ane_active_percent = (result.ane_power / ANE_MAX_POWER_WATTS * 100.0).min(100.0);
        }
        result
    }
}

fn energy_totals(
    functions: &IoReportFunctions,
    delta: &CFDictionary,
) -> EnergyTotals {
    let mut totals = EnergyTotals::default();
    for channel in decode_channels(functions, delta) {
        if channel.group != obfstr!("Energy Model") {
            continue;
        }
        accumulate_energy(&mut totals, &channel.name, joules(channel.integer_value, channel.unit.trim()));
    }
    totals
}

fn joules(
    energy: i64,
    unit: &str,
) -> f32 {
    let energy = energy as f32;
    let Some(prefix) = unit.trim().strip_suffix('J') else {
        return 0.0;
    };
    let scale = match prefix {
        "k" => 1e3,
        "" => 1.0,
        "m" => 1e-3,
        "u" | "µ" => 1e-6,
        "n" => 1e-9,
        "p" => 1e-12,
        _ => return 0.0,
    };
    energy * scale
}

fn accumulate_energy(
    totals: &mut EnergyTotals,
    name: &str,
    joules: f32,
) {
    if name == obfstr!("GPU Energy") {
        totals.gpu += joules;
    } else if name.ends_with(obfstr!("CPU Energy")) {
        totals.cpu += joules;
    } else if name.starts_with(obfstr!("ANE")) {
        totals.ane += joules;
    } else if name.starts_with(obfstr!("DRAM")) || name.starts_with(obfstr!("DCS")) || name.starts_with(obfstr!("AMCC")) {
        totals.ram += joules;
    } else if name.starts_with(obfstr!("GPU SRAM")) {
        totals.gpu_sram += joules;
    }
}

#[cfg(test)]
mod energy_dump {
    use std::collections::BTreeSet;

    use obfstr::obfstr;

    use super::{IoReport, decode_channels};

    #[test]
    #[ignore]
    fn dump_energy_model_channels() {
        println!("KEISOKU_ENERGY_BEGIN");
        let Some(report) = IoReport::new() else {
            println!("STATUS\tIOReport unavailable (subscription blocked or no permission)");
            println!("KEISOKU_ENERGY_END");
            return;
        };
        let Some(sample) = report.snapshot() else {
            println!("STATUS\tsnapshot unavailable");
            println!("KEISOKU_ENERGY_END");
            return;
        };
        let channels = decode_channels(report.functions, &sample.0);
        let mut groups = BTreeSet::new();
        let mut rails = 0;
        for channel in &channels {
            groups.insert(channel.group.clone());
            if channel.group == obfstr!("Energy Model") {
                rails += 1;
                println!("RAIL\t{}\t{}\t{}", channel.name, channel.unit, channel.integer_value);
            }
        }
        println!("STATUS\t{} energy-model rails, {} channels total", rails, channels.len());
        for group in &groups {
            println!("GROUP\t{}", group);
        }
        println!("KEISOKU_ENERGY_END");
    }
}
