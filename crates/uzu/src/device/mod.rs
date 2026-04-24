mod error;
mod os;

pub use error::DeviceError;
use os::{home_path, is_environment_sandboxed, is_keyring_available};
use serde::{Deserialize, Serialize};
use sysinfo::System;

#[bindings::export(ClassCloneable)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Device {
    pub os_name: Option<String>,
    pub cpu_name: Option<String>,
    pub memory_total: i64,
    pub home_path: String,
    pub is_environment_sandboxed: bool,
    pub is_keyring_available: bool,
}

impl Device {
    pub fn new() -> Result<Self, DeviceError> {
        let mut system_info = System::new_all();
        system_info.refresh_all();

        let os_name = System::long_os_version();
        let cpu_name = system_info.cpus().first().map(|cpu| cpu.brand().to_string());
        let memory_total = system_info.total_memory();
        let home_path = home_path().ok_or(DeviceError::UnsupportedDevice {})?;
        let is_environment_sandboxed = is_environment_sandboxed();
        let is_keyring_available = is_keyring_available();

        Ok(Self {
            os_name,
            cpu_name,
            memory_total: memory_total as i64,
            home_path: home_path.to_string_lossy().to_string(),
            is_environment_sandboxed,
            is_keyring_available,
        })
    }
}

#[bindings::export(Implementation)]
impl Device {
    #[bindings::export(Factory)]
    pub fn create() -> Result<Self, DeviceError> {
        Self::new()
    }
}
