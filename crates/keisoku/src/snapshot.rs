use serde::{Deserialize, Serialize};

use crate::{
    metrics::{
        BandwidthMetrics, BatteryMetrics, CpuMetrics, FanMetrics, GpuMetrics, MemoryMetrics, NeuralEngineMetrics,
        PowerMetrics, Temperatures, ThermalPressure,
    },
    sensor::Sensor,
    units::{Milliseconds, Watts},
};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub elapsed: Milliseconds,
    pub cpu: Option<CpuMetrics>,
    pub gpu: Option<GpuMetrics>,
    pub neural_engine: Option<NeuralEngineMetrics>,
    pub power: Option<PowerMetrics>,
    pub memory: Option<MemoryMetrics>,
    pub bandwidth: Option<BandwidthMetrics>,
    pub fans: Option<FanMetrics>,
    pub battery: Option<BatteryMetrics>,
    pub temperatures: Option<Temperatures>,
    pub thermal_pressure: Option<ThermalPressure>,
    pub sensors: Vec<Sensor>,
    /// HID voltage rails; on iOS the only un-gated electrical signal, empty elsewhere.
    #[serde(default)]
    pub voltage: Vec<Sensor>,
    /// HID current rails, paired with [`Snapshot::voltage`] by rail code.
    #[serde(default)]
    pub current: Vec<Sensor>,
}

impl Snapshot {
    /// Charger watts from matched HID voltage×current rails (e.g. `Charger VQ0u`×`IQ0u`); estimate, unlike the IOReport [`Snapshot::power`].
    pub fn rail_power(&self) -> Option<Watts> {
        use crate::component::Component;
        // Ceiling to discard mismatched-unit pairs; real charging stays far below.
        const MAX_PLAUSIBLE_WATTS: f64 = 1000.0;
        let split = |name: &str| name.rsplit_once(' ').map(|(area, code)| (area.to_owned(), code.to_owned()));
        let is_battery_rail = |sensor: &&Sensor| matches!(sensor.component, Component::Charger | Component::Battery);

        let mut best = 0f64;
        for volts in self.voltage.iter().filter(is_battery_rail) {
            let Some((varea, vcode)) = split(&volts.name) else {
                continue;
            };
            let Some(rail) = vcode.strip_prefix('V').filter(|rail| !rail.is_empty()) else {
                continue;
            };
            for amps in self.current.iter().filter(is_battery_rail) {
                let Some((aarea, acode)) = split(&amps.name) else {
                    continue;
                };
                if aarea == varea && acode.strip_prefix('I') == Some(rail) {
                    let watts = volts.value * amps.value;
                    if (0.0..=MAX_PLAUSIBLE_WATTS).contains(&watts) {
                        best = best.max(watts);
                    }
                }
            }
        }
        (best > 0.0).then_some(Watts(best as f32))
    }
}
