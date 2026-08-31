#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeviceIdentity {
    A14,
    A15,
    A16,
    A17Pro,
    A18,
    A18Pro,
    A19,
    A19Pro,
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
    M3Ultra,
    M4,
    M4Pro,
    M4Max,
    M5,
    M5Pro,
    M5Max,
    M5Ultra,
}

impl DeviceIdentity {
    fn from_name(device_name: &str) -> Self {
        let Some(chip) = device_name.strip_prefix("Apple ") else {
            panic!("unsupported Metal device {device_name:?}; add its DeviceIdentity mapping");
        };
        let chip = chip.strip_suffix(" GPU").unwrap_or(chip);
        match chip {
            "A14" => Self::A14,
            "A15" => Self::A15,
            "A16" => Self::A16,
            "A17 Pro" => Self::A17Pro,
            "A18" => Self::A18,
            "A18 Pro" => Self::A18Pro,
            "A19" => Self::A19,
            "A19 Pro" => Self::A19Pro,
            "M1" => Self::M1,
            "M1 Pro" => Self::M1Pro,
            "M1 Max" => Self::M1Max,
            "M1 Ultra" => Self::M1Ultra,
            "M2" => Self::M2,
            "M2 Pro" => Self::M2Pro,
            "M2 Max" => Self::M2Max,
            "M2 Ultra" => Self::M2Ultra,
            "M3" => Self::M3,
            "M3 Pro" => Self::M3Pro,
            "M3 Max" => Self::M3Max,
            "M3 Ultra" => Self::M3Ultra,
            "M4" => Self::M4,
            "M4 Pro" => Self::M4Pro,
            "M4 Max" => Self::M4Max,
            "M5" => Self::M5,
            "M5 Pro" => Self::M5Pro,
            "M5 Max" => Self::M5Max,
            "M5 Ultra" => Self::M5Ultra,
            _ => panic!("unsupported Metal device {device_name:?}; add its DeviceIdentity mapping"),
        }
    }

    const fn gpu_family(self) -> GpuFamily {
        match self {
            Self::A14 | Self::M1 | Self::M1Pro | Self::M1Max | Self::M1Ultra => GpuFamily::Legacy,
            Self::A15 | Self::A16 | Self::M2 | Self::M2Pro | Self::M2Max | Self::M2Ultra => GpuFamily::Apple8,
            Self::A17Pro
            | Self::A18
            | Self::A18Pro
            | Self::M3
            | Self::M3Pro
            | Self::M3Max
            | Self::M3Ultra
            | Self::M4
            | Self::M4Pro
            | Self::M4Max => GpuFamily::Apple9,
            Self::A19 | Self::A19Pro | Self::M5 | Self::M5Pro | Self::M5Max | Self::M5Ultra => GpuFamily::M5Plus,
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
#[path = "../../../unit/backends/metal/device_profile_test.rs"]
mod tests;
