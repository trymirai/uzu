#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GpuTuningTier {
    Legacy,
    Apple8,
    Apple9,
    Apple10,
}

impl GpuTuningTier {
    const fn from_capabilities(
        supports_apple8: bool,
        supports_apple9: bool,
        supports_apple10: bool,
    ) -> Self {
        if supports_apple10 {
            Self::Apple10
        } else if supports_apple9 {
            Self::Apple9
        } else if supports_apple8 {
            Self::Apple8
        } else {
            Self::Legacy
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
    tuning_tier: GpuTuningTier,
    size: DeviceSize,
    supports_mxu: bool,
}

impl DeviceProfile {
    pub(super) const fn new(
        tuning_tier: GpuTuningTier,
        size: DeviceSize,
        supports_mxu: bool,
    ) -> Self {
        Self {
            tuning_tier,
            size,
            supports_mxu,
        }
    }

    pub(super) const fn tuning_tier(self) -> GpuTuningTier {
        self.tuning_tier
    }

    pub(super) const fn size(self) -> DeviceSize {
        self.size
    }

    pub(super) const fn supports_mxu(self) -> bool {
        self.supports_mxu
    }
}

pub(super) fn classify_device(
    gpu_core_count: u32,
    supports_apple8: bool,
    supports_apple9: bool,
    supports_apple10: bool,
    supports_mxu: bool,
) -> DeviceProfile {
    DeviceProfile::new(
        GpuTuningTier::from_capabilities(supports_apple8, supports_apple9, supports_apple10),
        DeviceSize::from_gpu_core_count(gpu_core_count),
        supports_mxu,
    )
}

#[cfg(test)]
#[path = "../../../unit/backends/metal/device_profile_test.rs"]
mod tests;
