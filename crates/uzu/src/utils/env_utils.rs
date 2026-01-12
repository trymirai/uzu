#[derive(Copy, Clone, Debug)]
pub enum MetalEnvVar {
    DeviceWrapperType,
    CaptureFirstDecode,
    CaptureFirstPrefill,
    DebugMatmul,
}

impl MetalEnvVar {
    pub fn key(&self) -> &'static str {
        match self {
            MetalEnvVar::DeviceWrapperType => "METAL_DEVICE_WRAPPER_TYPE",
            MetalEnvVar::CaptureFirstDecode => "UZU_CAPTURE_FIRST_DECODE",
            MetalEnvVar::CaptureFirstPrefill => "UZU_CAPTURE_FIRST_PREFILL",
            MetalEnvVar::DebugMatmul => "UZU_DEBUG_MATMUL",
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

use std::sync::OnceLock;

static DEBUG_MATMUL_ENABLED: OnceLock<bool> = OnceLock::new();

pub fn debug_matmul_enabled() -> bool {
    *DEBUG_MATMUL_ENABLED.get_or_init(|| MetalEnvVar::DebugMatmul.is_enabled())
}
