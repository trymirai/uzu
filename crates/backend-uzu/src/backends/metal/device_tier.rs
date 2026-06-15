use metal::MTLDevice;
use objc2::runtime::ProtocolObject;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceTier {
    /// Small Apple9+ GPUs (M3/A17 and newer, below Max/Ultra class).
    SmallApple9,
    /// Small Apple8 GPUs (M2/A15/A16 generation).
    SmallApple8,
    /// Small pre-Apple8 GPUs (M1/A14 generation).
    SmallLegacy,
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
        DeviceTier::SmallApple9
    } else if supports_apple8_family {
        DeviceTier::SmallApple8
    } else {
        DeviceTier::SmallLegacy
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/backends/metal/device_tier_test.rs"]
mod tests;
