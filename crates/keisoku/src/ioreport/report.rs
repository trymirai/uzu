use std::time::{Duration, Instant};

use obfstr::obfstr;
use objc2_core_foundation::{CFDictionary, CFRetained};

use super::{
    IoReportFunctions, SocSample,
    bandwidth::{BandwidthAccumulator, accumulate_amc_bandwidth, accumulate_pmp},
    channel::decode_channels,
    energy_totals::EnergyTotals,
    frequency::{average_cluster_frequency, calculate_frequency, divide_or_zero},
    subscription::Subscription,
};
use crate::{EnergyModelChannel, soc::SocInfo, units::Watts};

pub struct IoReport {
    functions: &'static IoReportFunctions,
    subscription: Subscription,
}

pub(crate) struct RawEnergySample(CFRetained<CFDictionary>);

impl IoReport {
    pub fn new() -> Option<Self> {
        let functions = IoReportFunctions::get()?;
        let subscription = Subscription::new(functions)?;
        Some(Self {
            functions,
            subscription,
        })
    }

    pub(crate) fn energy_only() -> Option<Self> {
        let functions = IoReportFunctions::get()?;
        let subscription = Subscription::energy_model(functions)?;
        Some(Self {
            functions,
            subscription,
        })
    }

    pub(crate) fn snapshot(&self) -> Option<RawEnergySample> {
        self.subscription.snapshot(self.functions).map(RawEnergySample)
    }

    pub(crate) fn cumulative_energy(&self) -> Option<EnergyTotals> {
        let sample = self.snapshot()?;
        Some(energy_totals(self.functions, &sample.0))
    }

    pub(crate) fn energy_model_channels(&self) -> Box<[EnergyModelChannel]> {
        let Some(sample) = self.snapshot() else {
            return Box::default();
        };
        decode_channels(self.functions, &sample.0)
            .into_iter()
            .filter(|channel| channel.group == obfstr!("Energy Model"))
            .map(|channel| EnergyModelChannel {
                name: channel.name,
                unit: channel.unit,
                value: channel.integer_value,
            })
            .collect()
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
                energy.accumulate(&channel.name, channel.integer_value, &channel.unit);
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
        let package = Watts((energy.total() / elapsed.as_secs_f64().max(0.001)) as f32);
        let power = energy.power_metrics(elapsed, package);
        result.cpu_power = power.cpu.value();
        result.gpu_power = power.gpu.value();
        result.gpu_ram_power = power.gpu_sram.value();
        result.ane_power = power.ane.value();
        result.ram_power = power.ram.value();
        result.total_power = power.total().value();

        bandwidth.finish(window_milliseconds, &mut result);
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
        totals.accumulate(&channel.name, channel.integer_value, &channel.unit);
    }
    totals
}
