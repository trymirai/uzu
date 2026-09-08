use anyhow::Result;
use keisoku::Device as MetricsDevice;
use uzu::device::Device as HostDevice;

/// Host facts recorded on every row so rows from different machines stay comparable.
pub struct DeviceInfo {
    pub os: String,
    pub chip: String,
    pub ram_total_bytes: u64,
    pub gpu_cores: u8,
}

impl DeviceInfo {
    pub fn collect() -> Result<Self> {
        let mut device = MetricsDevice::new();
        Ok(Self {
            os: HostDevice::new()?.os_name.unwrap_or_default(),
            chip: device.chip(),
            ram_total_bytes: device.memory().map(|memory| memory.ram_total.value()).unwrap_or_default(),
            gpu_cores: device.gpu_cores(),
        })
    }
}
