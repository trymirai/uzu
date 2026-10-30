mod error;
mod os;

pub use error::DeviceError;
pub(crate) use os::home_path;
use serde::{Deserialize, Serialize};
#[cfg(not(target_family = "wasm"))]
use sysinfo::System;

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Device {
    pub os_name: Option<String>,
    pub cpu_name: Option<String>,
    pub memory_total: Option<i64>,
    pub home_path: String,
}

impl Device {
    #[cfg(not(target_family = "wasm"))]
    pub fn new() -> Result<Self, DeviceError> {
        let mut system_info = System::new_all();
        system_info.refresh_all();

        let os_name = System::long_os_version();
        let cpu_name = system_info.cpus().first().map(|cpu| cpu.brand().to_string());
        let memory_total = system_info.total_memory();
        let home_path = home_path().ok_or(DeviceError::UnsupportedDevice {})?;

        Ok(Self {
            os_name,
            cpu_name,
            memory_total: Some(memory_total as i64),
            home_path: home_path.to_string_lossy().to_string(),
        })
    }

    #[cfg(target_family = "wasm")]
    pub fn new() -> Result<Self, DeviceError> {
        let window = web_sys::window().ok_or(DeviceError::UnsupportedDevice {})?;
        let navigator = window.navigator();
        let memory_gib =
            web_sys::js_sys::Reflect::get(&navigator, &"deviceMemory".into()).ok().and_then(|value| value.as_f64());
        Ok(Self {
            os_name: None,
            cpu_name: None,
            memory_total: memory_gib.map(|memory| (memory * 1024.0_f64.powi(3)) as i64),
            home_path: home_path().unwrap().to_string_lossy().into_owned(),
        })
    }
}

#[bindings::export(Implementation)]
impl Device {
    #[bindings::export(Method(Factory))]
    pub fn create() -> Result<Self, DeviceError> {
        Self::new()
    }
}
