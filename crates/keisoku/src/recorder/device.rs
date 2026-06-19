use serde::{Deserialize, Serialize};

use crate::{Collector, units::Bytes};

/// Static description of the machine the session was recorded on.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Device {
    pub os: String,
    pub chip: String,
    pub ram_total: Bytes,
    pub efficiency_cores: u8,
    pub performance_cores: u8,
    pub gpu_cores: u8,
}

impl Device {
    /// Builds the device description, enriching with SoC details on macOS.
    pub(crate) fn detect(#[allow(unused_variables)] collector: &Collector) -> Device {
        let os = sysinfo::System::long_os_version().unwrap_or_default();
        let mut system = sysinfo::System::new_all();
        system.refresh_all();
        let chip = system.cpus().first().map(|c| c.brand().trim().to_string()).unwrap_or_default();
        let ram_total = Bytes(system.total_memory());

        #[cfg(target_os = "macos")]
        if let Some(soc) = collector.soc() {
            return Device {
                os,
                chip: if soc.chip_name.is_empty() {
                    chip
                } else {
                    soc.chip_name.clone()
                },
                ram_total,
                efficiency_cores: soc.ecpu_cores,
                performance_cores: soc.pcpu_cores,
                gpu_cores: soc.gpu_cores,
            };
        }

        Device {
            os,
            chip,
            ram_total,
            ..Default::default()
        }
    }
}
