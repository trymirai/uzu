use metal::MTLDevice;
use objc2::runtime::ProtocolObject;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceTier {
    /// Small Apple9+ GPUs (M3/A17 and newer, below Max/Ultra class).
    Small,
    /// Small Apple8 GPUs (M2/A15/A16 generation).
    SmallG14,
    /// Small pre-Apple8 GPUs (M1/A14 generation).
    SmallG13,
    /// Max/Ultra-class GPUs, currently classified by >= 30 GPU cores.
    Large,
}

pub(crate) fn device_tier_for_device(
    gpu_core_count: u32,
    device: &ProtocolObject<dyn MTLDevice>,
) -> DeviceTier {
    device_tier_for(
        gpu_core_count,
        device.supports_family(metal::MTLGPUFamily::Apple8),
        device.supports_family(metal::MTLGPUFamily::Apple9),
    )
}

fn device_tier_for(
    gpu_core_count: u32,
    supports_apple8_family: bool,
    supports_apple9_family: bool,
) -> DeviceTier {
    if gpu_core_count >= 30 {
        DeviceTier::Large
    } else if supports_apple9_family {
        DeviceTier::Small
    } else if supports_apple8_family {
        DeviceTier::SmallG14
    } else {
        DeviceTier::SmallG13
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uzu_test;

    #[uzu_test]
    fn device_tier_detection() {
        assert_eq!(device_tier_for(5, true, true), DeviceTier::Small); // A18 Pro
        assert_eq!(device_tier_for(8, false, false), DeviceTier::SmallG13); // M1
        assert_eq!(device_tier_for(10, true, false), DeviceTier::SmallG14); // M2
        assert_eq!(device_tier_for(19, true, false), DeviceTier::SmallG14); // M2 Pro
        assert_eq!(device_tier_for(20, true, true), DeviceTier::Small); // M4 Pro
        assert_eq!(device_tier_for(20, false, true), DeviceTier::Small); // Prefer newest supported family
        assert_eq!(device_tier_for(40, true, true), DeviceTier::Large); // M3/M4 Max
        assert_eq!(device_tier_for(32, false, false), DeviceTier::Large); // M1 Max stays large
    }
}
