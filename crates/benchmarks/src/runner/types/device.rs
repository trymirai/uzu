use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Device {
    pub os_name: Option<String>,
    pub cpu_name: Option<String>,
    pub memory_total: u64,
}
