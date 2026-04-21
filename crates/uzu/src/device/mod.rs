mod error;
mod os;

use std::path::PathBuf;

pub use error::Error;
use os::{home_path, is_environment_sandboxed};
use serde::{Deserialize, Serialize};
use sysinfo::System;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Device {
    pub os_name: Option<String>,
    pub cpu_name: Option<String>,
    pub memory_total: u64,
    pub home_path: PathBuf,
    pub is_environment_sandboxed: bool,
}

impl Device {
    pub fn new() -> Result<Self, Error> {
        let mut system_info = System::new_all();
        system_info.refresh_all();

        let os_name = System::long_os_version();
        let cpu_name = system_info.cpus().first().map(|cpu| cpu.brand().to_string());
        let memory_total = system_info.total_memory();
        let home_path = home_path().ok_or(Error::UnsupportedDevice)?;
        let is_environment_sandboxed = is_environment_sandboxed();

        Ok(Self {
            os_name,
            cpu_name,
            memory_total,
            home_path,
            is_environment_sandboxed,
        })
    }
}
