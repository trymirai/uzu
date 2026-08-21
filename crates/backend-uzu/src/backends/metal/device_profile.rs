#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeviceIdentity {
    M1,
    M2,
    M2Pro,
    M3Max,
    M4,
    M4Pro,
    M5Max,
    Other,
}

impl DeviceIdentity {
    fn from_name(device_name: &str) -> Self {
        match device_name {
            "Apple M1" => Self::M1,
            "Apple M2" => Self::M2,
            "Apple M2 Pro" => Self::M2Pro,
            "Apple M3 Max" => Self::M3Max,
            "Apple M4" => Self::M4,
            "Apple M4 Pro" => Self::M4Pro,
            "Apple M5 Max" => Self::M5Max,
            _ => Self::Other,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GpuFamily {
    Legacy,
    Apple8,
    Apple9,
    M5Plus,
}

impl GpuFamily {
    fn classify(
        device_name: &str,
        supports_apple8_family: bool,
        supports_apple9_family: bool,
    ) -> Self {
        let generation = device_name.split_whitespace().find_map(|part| part.strip_prefix('M')?.parse::<u32>().ok());
        match generation {
            Some(5..) => Self::M5Plus,
            Some(3..=4) => Self::Apple9,
            Some(2) => Self::Apple8,
            Some(1) => Self::Legacy,
            _ if supports_apple9_family => Self::Apple9,
            _ if supports_apple8_family => Self::Apple8,
            _ => Self::Legacy,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeviceSize {
    Small,
    Large,
}

impl DeviceSize {
    fn from_gpu_core_count(gpu_core_count: u32) -> Self {
        if gpu_core_count >= 30 {
            Self::Large
        } else {
            Self::Small
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DeviceProfile {
    identity: DeviceIdentity,
    gpu_family: GpuFamily,
    size: DeviceSize,
    supports_mxu: bool,
}

impl DeviceProfile {
    pub(super) const fn new(
        identity: DeviceIdentity,
        gpu_family: GpuFamily,
        size: DeviceSize,
        supports_mxu: bool,
    ) -> Self {
        Self {
            identity,
            gpu_family,
            size,
            supports_mxu,
        }
    }

    pub(super) const fn identity(self) -> DeviceIdentity {
        self.identity
    }

    pub(super) const fn gpu_family(self) -> GpuFamily {
        self.gpu_family
    }

    pub(super) const fn size(self) -> DeviceSize {
        self.size
    }

    pub(super) const fn supports_mxu(self) -> bool {
        self.supports_mxu
    }
}

pub(super) fn classify_device(
    device_name: &str,
    gpu_core_count: u32,
    supports_apple8_family: bool,
    supports_apple9_family: bool,
    supports_mxu: bool,
) -> DeviceProfile {
    DeviceProfile::new(
        DeviceIdentity::from_name(device_name),
        GpuFamily::classify(device_name, supports_apple8_family, supports_apple9_family),
        DeviceSize::from_gpu_core_count(gpu_core_count),
        supports_mxu,
    )
}

#[cfg(test)]
#[path = "../../../tests/unit/backends/metal/device_profile_test.rs"]
mod tests;
