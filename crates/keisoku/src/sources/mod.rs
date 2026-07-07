mod deferred;

use deferred::Deferred;

use crate::{
    client::SensorReader,
    sensor::{Sensor, SensorKind},
};
#[cfg(target_os = "macos")]
use crate::{decode::FrequencyTables, smc::Smc, soc::SocInfo};

pub struct Sources {
    system: Deferred<sysinfo::System>,
    temperature: Deferred<Option<SensorReader>>,
    voltage: Deferred<Option<SensorReader>>,
    current: Deferred<Option<SensorReader>>,
    // Built through a shared `&` (OnceCell) so several interval-metric contexts
    // can borrow the SoC tables at once without a `&mut` conflict.
    #[cfg(target_os = "macos")]
    soc: std::cell::OnceCell<Option<SocInfo>>,
    #[cfg(target_os = "macos")]
    smc: std::cell::OnceCell<Option<Smc>>,
}

impl Sources {
    pub fn new() -> Self {
        Self {
            system: Deferred::new(build_system),
            temperature: Deferred::new(|| SensorReader::new(SensorKind::Temperature)),
            voltage: Deferred::new(|| SensorReader::new(SensorKind::Voltage)),
            current: Deferred::new(|| SensorReader::new(SensorKind::Current)),
            #[cfg(target_os = "macos")]
            soc: std::cell::OnceCell::new(),
            #[cfg(target_os = "macos")]
            smc: std::cell::OnceCell::new(),
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
    pub(crate) fn soc(&self) -> Option<&SocInfo> {
        self.soc.get_or_init(SocInfo::new).as_ref()
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn smc(&self) -> Option<&Smc> {
        self.smc.get_or_init(Smc::new).as_ref()
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn frequencies(&self) -> FrequencyTables<'_> {
        match self.soc() {
            Some(soc) => FrequencyTables {
                ecpu: &soc.ecpu_frequencies,
                pcpu: &soc.pcpu_frequencies,
                gpu: &soc.gpu_frequencies,
                ecpu_cores: soc.ecpu_cores,
                pcpu_cores: soc.pcpu_cores,
            },
            None => FrequencyTables::default(),
        }
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
