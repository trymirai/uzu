#[derive(Copy, Clone, Debug)]
pub enum MetalEnvVar {
    DeviceWrapperType,
}

impl MetalEnvVar {
    pub fn key(&self) -> &'static str {
        match self {
            MetalEnvVar::DeviceWrapperType => "METAL_DEVICE_WRAPPER_TYPE",
        }
    }

    pub fn value(&self) -> String {
        std::env::var(self.key()).unwrap_or_default()
    }

    pub fn is_enabled(&self) -> bool {
        let upper = self.value().to_ascii_uppercase();
        matches!(upper.as_str(), "1" | "YES" | "TRUE")
    }
}
