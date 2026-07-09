mod battery;
mod deferred;
#[cfg(target_os = "macos")]
pub(crate) mod interval;
mod memory;
mod rail_power;
mod sensors;
#[cfg(target_os = "macos")]
mod smc;
#[cfg(target_os = "macos")]
mod soc;
mod thermal;

use deferred::Deferred;

#[cfg(target_os = "macos")]
use crate::metrics::Fan;
#[cfg(target_os = "macos")]
use crate::sys::{smc::Smc, soc::SocInfo};
use crate::{
    metrics::FanMetrics,
    sensor::{Sensor, SensorKind},
    sys::hid::SensorReader,
    units::Watts,
};

pub(crate) struct Sources {
    temperature: Deferred<Option<SensorReader>>,
    voltage: Deferred<Option<SensorReader>>,
    current: Deferred<Option<SensorReader>>,
    #[cfg(target_os = "macos")]
    soc: std::cell::OnceCell<Option<SocInfo>>,
    #[cfg(target_os = "macos")]
    smc: std::cell::OnceCell<Option<Smc>>,
}

impl Sources {
    pub fn new() -> Self {
        Self {
            temperature: Deferred::new(|| sensors::new_reader(SensorKind::Temperature)),
            voltage: Deferred::new(|| sensors::new_reader(SensorKind::Voltage)),
            current: Deferred::new(|| sensors::new_reader(SensorKind::Current)),
            #[cfg(target_os = "macos")]
            soc: std::cell::OnceCell::new(),
            #[cfg(target_os = "macos")]
            smc: std::cell::OnceCell::new(),
        }
    }

    pub(crate) fn chip(&self) -> String {
        #[cfg(target_os = "macos")]
        {
            self.soc().map(|soc| soc.chip_name.clone()).unwrap_or_default()
        }
        #[cfg(not(target_os = "macos"))]
        {
            crate::sys::sysctl_string("hw.machine").unwrap_or_default()
        }
    }

    pub(crate) fn efficiency_cores(&self) -> u8 {
        #[cfg(target_os = "macos")]
        {
            self.soc().map(|soc| soc.ecpu_cores).unwrap_or(0)
        }
        #[cfg(not(target_os = "macos"))]
        {
            crate::sys::perflevel_cores().1
        }
    }

    pub(crate) fn performance_cores(&self) -> u8 {
        #[cfg(target_os = "macos")]
        {
            self.soc().map(|soc| soc.pcpu_cores).unwrap_or(0)
        }
        #[cfg(not(target_os = "macos"))]
        {
            crate::sys::perflevel_cores().0
        }
    }

    pub(crate) fn gpu_cores(&self) -> u8 {
        #[cfg(target_os = "macos")]
        {
            self.soc().map(|soc| soc.gpu_cores).unwrap_or(0)
        }
        #[cfg(not(target_os = "macos"))]
        {
            0
        }
    }

    pub(crate) fn temperature_sensors(&mut self) -> Box<[Sensor]> {
        self.temperature.get().as_mut().map(sensors::read_reader).unwrap_or_default()
    }

    pub(crate) fn voltage_sensors(&mut self) -> Box<[Sensor]> {
        self.voltage.get().as_mut().map(sensors::read_reader).unwrap_or_default()
    }

    pub(crate) fn current_sensors(&mut self) -> Box<[Sensor]> {
        self.current.get().as_mut().map(sensors::read_reader).unwrap_or_default()
    }

    pub(crate) fn memory(&mut self) -> Option<crate::metrics::MemoryMetrics> {
        memory::read_memory()
    }

    pub(crate) fn battery(&mut self) -> Option<crate::metrics::BatteryMetrics> {
        battery::read_battery()
    }

    pub(crate) fn thermal(&mut self) -> Option<crate::metrics::ThermalPressure> {
        thermal::read_thermal()
    }

    pub(crate) fn rail_power(&mut self) -> Option<Watts> {
        let voltage = self.voltage_sensors();
        let current = self.current_sensors();
        rail_power::rail_power(&voltage, &current)
    }

    pub(crate) fn fans(&self) -> Option<FanMetrics> {
        #[cfg(target_os = "macos")]
        {
            self.smc().map(|smc| {
                let snapshot = smc::fans(smc);
                let fans = snapshot
                    .fans
                    .into_iter()
                    .map(|fan| Fan {
                        actual: fan.actual,
                        minimum: fan.minimum,
                        maximum: fan.maximum,
                        target: fan.target,
                    })
                    .collect();
                FanMetrics {
                    fans,
                }
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn soc(&self) -> Option<&SocInfo> {
        self.soc.get_or_init(soc::new_soc).as_ref()
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn smc(&self) -> Option<&Smc> {
        self.smc.get_or_init(smc::new_smc).as_ref()
    }
}

impl Default for Sources {
    fn default() -> Self {
        Self::new()
    }
}
