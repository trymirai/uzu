//! `keisoku` (計測, "measurement") — system telemetry for Apple platforms.
//!
//! Collects CPU/GPU utilization, power (W), RAM/swap, and per-sensor
//! temperatures, and records them as a time-series [`Session`] for correlating
//! against workloads (e.g. model inference). Private Apple APIs (IOReport, IOHID)
//! are bound through `kanka` (obfuscated dlsym); the rest is `libc`/`sysinfo`.

mod collector;
mod component;
mod metrics;
mod recorder;
mod sensor;
mod snapshot;
mod units;

#[cfg(target_os = "macos")]
mod cf;
#[cfg(target_vendor = "apple")]
mod client;
#[cfg(target_os = "macos")]
mod cpu_load;
#[cfg(target_os = "macos")]
mod ioreport;
#[cfg(target_os = "macos")]
mod soc;
#[cfg(target_vendor = "apple")]
mod sys;

pub use collector::Collector;
pub use component::{Component, classify};
pub use metrics::{
    BandwidthMetrics, CpuMetrics, GpuMetrics, MemoryMetrics, NeuralEngineMetrics, PowerMetrics, Temperatures,
    ThermalPressure,
};
pub use recorder::{Config, Device, Marker, RecorderHandle, Session, start};
pub use sensor::{Sensor, SensorKind, current_sensors, thermal_sensors, voltage_sensors};
pub use snapshot::Snapshot;
pub use units::{Bytes, Celsius, GigabytesPerSecond, Megahertz, Milliseconds, Percent, Watts};

/// Reads every sensor of `kind` (temperature/voltage/current). Empty off Apple.
#[cfg(target_vendor = "apple")]
pub fn sensors(kind: SensorKind) -> Vec<Sensor> {
    client::collect(kind)
}

/// Whether the private IOHID sensor API resolved on this system.
#[cfg(target_vendor = "apple")]
pub fn sensors_available() -> bool {
    client::is_available()
}

#[cfg(not(target_vendor = "apple"))]
pub fn sensors(_kind: SensorKind) -> Vec<Sensor> {
    Vec::new()
}

#[cfg(not(target_vendor = "apple"))]
pub fn sensors_available() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensor_queries_do_not_panic() {
        let _ = sensors_available();
        for kind in [SensorKind::Temperature, SensorKind::Voltage, SensorKind::Current] {
            for sensor in sensors(kind) {
                assert_eq!(sensor.kind, kind);
                assert!(sensor.value.is_finite());
            }
        }
    }

    #[test]
    fn one_snapshot_is_well_formed() {
        let mut collector = Collector::new();
        let snapshot = collector.sample(std::time::Duration::from_millis(120));
        if let Some(power) = &snapshot.power {
            assert!(power.total.value().is_finite() && power.total.value() >= 0.0);
        }
        if let Some(memory) = &snapshot.memory {
            assert!(memory.ram_total >= memory.ram_usage);
        }
    }
}
