use std::time::Duration;

use obfstr::obfstr;

use crate::{
    EnergyMetrics, PowerMetrics,
    units::{Joules, Watts},
};

#[derive(Default, Clone, Copy)]
pub(crate) struct EnergyTotals {
    cpu: f64,
    gpu: f64,
    gpu_sram: f64,
    ane: f64,
    ram: f64,
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

    pub(crate) fn since(
        &self,
        earlier: &Self,
    ) -> Self {
        Self {
            cpu: self.cpu - earlier.cpu,
            gpu: self.gpu - earlier.gpu,
            gpu_sram: self.gpu_sram - earlier.gpu_sram,
            ane: self.ane - earlier.ane,
            ram: self.ram - earlier.ram,
        }
    }

    pub(crate) fn total(&self) -> f64 {
        self.cpu + self.gpu + self.ane + self.ram
    }

    pub(crate) fn energy_metrics(
        &self,
        package: Joules,
    ) -> EnergyMetrics {
        EnergyMetrics {
            cpu: Joules(self.cpu as f32),
            gpu: Joules(self.gpu as f32),
            gpu_sram: Joules(self.gpu_sram as f32),
            ane: Joules(self.ane as f32),
            ram: Joules(self.ram as f32),
            package,
        }
    }

    pub(crate) fn power_metrics(
        &self,
        elapsed: Duration,
        package: Watts,
    ) -> PowerMetrics {
        let elapsed_secs = elapsed.as_secs_f64().max(0.001);
        PowerMetrics {
            cpu: Watts((self.cpu / elapsed_secs) as f32),
            gpu: Watts((self.gpu / elapsed_secs) as f32),
            gpu_sram: Watts((self.gpu_sram / elapsed_secs) as f32),
            ane: Watts((self.ane / elapsed_secs) as f32),
            ram: Watts((self.ram / elapsed_secs) as f32),
            package,
        }
    }
}

fn joules(
    energy: i64,
    unit: &str,
) -> f64 {
    let energy = energy as f64;
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
