#[derive(Copy, Clone, Debug)]
pub enum EnvVar {
    DeviceWrapperType,
    CaptureFirstDecode,
    CaptureFirstPrefill,
}

impl EnvVar {
    pub fn key(&self) -> &'static str {
        match self {
            EnvVar::DeviceWrapperType => "UZU_DEVICE_WRAPPER_TYPE",
            EnvVar::CaptureFirstDecode => "UZU_CAPTURE_FIRST_DECODE",
            EnvVar::CaptureFirstPrefill => "UZU_CAPTURE_FIRST_PREFILL",
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
