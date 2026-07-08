mod battery;
mod deferred;
pub(crate) mod interval;
mod memory;
mod rail_power;
mod sensors;
mod smc;
#[cfg(target_os = "macos")]
mod soc;
mod thermal;

use deferred::Deferred;

#[cfg(target_os = "macos")]
use crate::sys::{ioreport::decode::FrequencyTables, smc::Smc, soc::SocInfo};
use crate::{
    providers::data::{Fan, FanMetrics},
    sensor::{Sensor, SensorKind},
    sys::hid::SensorReader,
    units::Watts,
};

pub struct Sources {
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

    pub(crate) fn temperature_sensors(&mut self) -> Box<[Sensor]> {
        self.temperature.get().as_mut().map(sensors::read_reader).unwrap_or_default()
    }

    pub(crate) fn voltage_sensors(&mut self) -> Box<[Sensor]> {
        self.voltage.get().as_mut().map(sensors::read_reader).unwrap_or_default()
    }

    pub(crate) fn current_sensors(&mut self) -> Box<[Sensor]> {
        self.current.get().as_mut().map(sensors::read_reader).unwrap_or_default()
    }

    pub(crate) fn memory(&mut self) -> Option<crate::providers::data::MemoryMetrics> {
        memory::read_memory()
    }

    pub(crate) fn battery(&mut self) -> Option<crate::providers::data::BatteryMetrics> {
        battery::read_battery()
    }

    pub(crate) fn thermal(&mut self) -> Option<crate::providers::data::ThermalPressure> {
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

    #[cfg(target_os = "macos")]
    pub(crate) fn frequencies(&self) -> FrequencyTables<'_> {
        match self.soc() {
            Some(soc) => soc::frequencies(soc),
            None => FrequencyTables::default(),
        }
    }
}

impl Default for Sources {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn collect_sensors(kind: SensorKind) -> Box<[Sensor]> {
    sensors::collect(kind)
}
