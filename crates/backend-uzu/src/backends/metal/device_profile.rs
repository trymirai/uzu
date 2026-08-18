#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceSize {
    /// Below Max/Ultra class: base and Pro dies, plus Max parts binned under
    /// the core-count cutoff (a 24-core M1 Max lands here, a 32-core one does not).
    Small,
    /// Max/Ultra-class GPUs, currently classified by >= 30 GPU cores.
    Large,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceGeneration {
    Legacy, // M1 (G13) and A14 or older
    Apple8, // M2, A15/16
    Apple9, // M3/4, A17 Pro, A18
    M5Plus, // M5 (Appl9 + MXU support)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceProfile {
    size: DeviceSize,
    generation: DeviceGeneration,
}

impl DeviceProfile {
    pub const fn new(
        size: DeviceSize,
        generation: DeviceGeneration,
    ) -> Self {
        Self {
            size,
            generation,
        }
    }

    pub const fn size(self) -> DeviceSize {
        self.size
    }

    pub const fn generation(self) -> DeviceGeneration {
        self.generation
    }
}

pub(super) fn classify_device(
    gpu_core_count: u32,
    supports_apple8_family: bool,
    supports_apple9_family: bool,
    supports_mxu: bool,
) -> DeviceProfile {
    let size = if gpu_core_count >= 30 {
        DeviceSize::Large
    } else {
        DeviceSize::Small
    };
    // MXU is probed first: M5 also reports Apple9, so a family check alone
    // cannot separate the two generations.
    let generation = if supports_mxu {
        DeviceGeneration::M5Plus
    } else if supports_apple9_family {
        DeviceGeneration::Apple9
    } else if supports_apple8_family {
        DeviceGeneration::Apple8
    } else {
        DeviceGeneration::Legacy
    };
    DeviceProfile::new(size, generation)
}

#[cfg(test)]
#[path = "../../../tests/unit/backends/metal/device_profile_test.rs"]
mod tests;
