#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeviceIdentity {
    M1,
    M1Pro,
    M1Max,
    M1Ultra,
    M2,
    M2Pro,
    M2Max,
    M2Ultra,
    M3,
    M3Pro,
    M3Max,
    M4,
    M4Pro,
    M4Max,
    M5,
    M5Pro,
    M5Max,
}

impl DeviceIdentity {
    fn from_name(device_name: &str) -> Self {
        let Some(chip) = device_name.strip_prefix("Apple M") else {
            panic!("unsupported Metal device {device_name:?}; add its DeviceIdentity mapping");
        };
        match chip {
            "1" => Self::M1,
            "1 Pro" => Self::M1Pro,
            "1 Max" => Self::M1Max,
            "1 Ultra" => Self::M1Ultra,
            "2" => Self::M2,
            "2 Pro" => Self::M2Pro,
            "2 Max" => Self::M2Max,
            "2 Ultra" => Self::M2Ultra,
            "3" => Self::M3,
            "3 Pro" => Self::M3Pro,
            "3 Max" => Self::M3Max,
            "4" => Self::M4,
            "4 Pro" => Self::M4Pro,
            "4 Max" => Self::M4Max,
            "5" => Self::M5,
            "5 Pro" => Self::M5Pro,
            "5 Max" => Self::M5Max,
            _ => panic!("unsupported Metal device {device_name:?}; add its DeviceIdentity mapping"),
        }
    }

    const fn gpu_family(self) -> GpuFamily {
        match self {
            Self::M1 | Self::M1Pro | Self::M1Max | Self::M1Ultra => GpuFamily::Legacy,
            Self::M2 | Self::M2Pro | Self::M2Max | Self::M2Ultra => GpuFamily::Apple8,
            Self::M3 | Self::M3Pro | Self::M3Max | Self::M4 | Self::M4Pro | Self::M4Max => GpuFamily::Apple9,
            Self::M5 | Self::M5Pro | Self::M5Max => GpuFamily::M5Plus,
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
    pub(super) fn contains(
        self,
        identity: DeviceIdentity,
    ) -> bool {
        identity.gpu_family() == self
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
    size: DeviceSize,
    supports_mxu: bool,
}

impl DeviceProfile {
    pub(super) const fn new(
        identity: DeviceIdentity,
        size: DeviceSize,
        supports_mxu: bool,
    ) -> Self {
        Self {
            identity,
            size,
            supports_mxu,
        }
    }

    pub(super) const fn identity(self) -> DeviceIdentity {
        self.identity
    }

    pub(super) const fn gpu_family(self) -> GpuFamily {
        self.identity.gpu_family()
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
    supports_mxu: bool,
) -> DeviceProfile {
    DeviceProfile::new(
        DeviceIdentity::from_name(device_name),
        DeviceSize::from_gpu_core_count(gpu_core_count),
        supports_mxu,
    )
}

#[cfg(test)]
#[path = "../../../tests/unit/backends/metal/device_profile_test.rs"]
mod tests;
