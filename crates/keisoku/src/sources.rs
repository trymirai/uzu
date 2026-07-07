use crate::{
    client::SensorReader,
    sensor::{Sensor, SensorKind},
};
#[cfg(target_os = "macos")]
use crate::{smc::Smc, soc::SocInfo};

struct Lazy<T> {
    slot: Option<T>,
    init: fn() -> T,
}

impl<T> Lazy<T> {
    const fn new(init: fn() -> T) -> Self {
        Self {
            slot: None,
            init,
        }
    }

    fn get(&mut self) -> &mut T {
        let init = self.init;
        self.slot.get_or_insert_with(init)
    }
}

pub struct Sources {
    system: Lazy<sysinfo::System>,
    temperature: Lazy<Option<SensorReader>>,
    voltage: Lazy<Option<SensorReader>>,
    current: Lazy<Option<SensorReader>>,
    #[cfg(target_os = "macos")]
    soc: Lazy<Option<SocInfo>>,
    #[cfg(target_os = "macos")]
    smc: Lazy<Option<Smc>>,
}

impl Sources {
    pub fn new() -> Self {
        Self {
            system: Lazy::new(build_system),
            temperature: Lazy::new(|| SensorReader::new(SensorKind::Temperature)),
            voltage: Lazy::new(|| SensorReader::new(SensorKind::Voltage)),
            current: Lazy::new(|| SensorReader::new(SensorKind::Current)),
            #[cfg(target_os = "macos")]
            soc: Lazy::new(SocInfo::new),
            #[cfg(target_os = "macos")]
            smc: Lazy::new(Smc::new),
        }
    }

    pub(crate) fn system(&mut self) -> &sysinfo::System {
        self.system.get()
    }

    pub(crate) fn temperature_sensors(&mut self) -> Box<[Sensor]> {
        self.temperature.get().as_mut().map(SensorReader::read).unwrap_or_default()
    }

    pub(crate) fn voltage_sensors(&mut self) -> Box<[Sensor]> {
        self.voltage.get().as_mut().map(SensorReader::read).unwrap_or_default()
    }

    pub(crate) fn current_sensors(&mut self) -> Box<[Sensor]> {
        self.current.get().as_mut().map(SensorReader::read).unwrap_or_default()
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn soc(&mut self) -> Option<&SocInfo> {
        self.soc.get().as_ref()
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn smc(&mut self) -> Option<&Smc> {
        self.smc.get().as_ref()
    }
}

impl Default for Sources {
    fn default() -> Self {
        Self::new()
    }
}

fn build_system() -> sysinfo::System {
    let mut system = sysinfo::System::new_all();
    system.refresh_all();
    system
}
