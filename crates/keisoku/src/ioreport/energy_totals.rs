use std::time::Duration;

use obfstr::obfstr;

use crate::{
    EnergyMetrics, PowerMetrics,
    units::{Joules, Watts},
};

#[derive(Default)]
pub(crate) struct EnergyTotals {
    cpu: f32,
    gpu: f32,
    gpu_sram: f32,
    ane: f32,
    ram: f32,
}

impl EnergyTotals {
    pub(super) fn accumulate(
        &mut self,
        name: &str,
        value: i64,
        unit: &str,
    ) {
        let joules = joules(value, unit);
        if name == obfstr!("GPU Energy") {
            self.gpu += joules;
        } else if name.ends_with(obfstr!("CPU Energy")) {
            self.cpu += joules;
        } else if name.starts_with(obfstr!("ANE")) {
            self.ane += joules;
        } else if name.starts_with(obfstr!("DRAM")) || name.starts_with(obfstr!("DCS")) || name.starts_with(obfstr!("AMCC")) {
            self.ram += joules;
        } else if name.starts_with(obfstr!("GPU SRAM")) {
            self.gpu_sram += joules;
        }
    }

    pub(crate) fn total(&self) -> f32 {
        self.cpu + self.gpu + self.ane + self.ram
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
            package,
        }
    }
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
