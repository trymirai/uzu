use byte_unit::Byte;
use metal::{MTLDevice, MTLDeviceExt as _};
use objc2::runtime::ProtocolObject;

use super::metal_extensions::{DeviceExt, DeviceGeneration, GpuTier};
use crate::backends::common::DeviceCapabilities;

#[derive(Debug, Clone)]
pub struct MetalDeviceCapabilities {
    pub generation: DeviceGeneration,
    pub tier: GpuTier,
    pub family_name: String,
    pub gpu_core_count: u32,
    pub max_threadgroup_memory: Byte,
    pub shared_memory_size: Byte,
    pub supports_simd_group: bool,
    pub supports_simd_group_matrix: bool,
    pub supports_simd_reduction: bool,
    pub supports_simd_shuffle_and_fill: bool,
    pub supports_simd_shuffles_and_broadcast: bool,
    pub supports_mxu: bool,
    pub supports_tls: bool,
}

impl DeviceCapabilities for MetalDeviceCapabilities {}

impl MetalDeviceCapabilities {
    pub fn from_device(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        let architecture_name = device.architecture().name();
        let (generation, tier) = Self::parse_architecture(&architecture_name);
        let max_threadgroup_memory = Byte::from_u64(device.max_threadgroup_memory_length() as u64);

        Self {
            generation,
            tier,
            family_name: device.family_name(),
            gpu_core_count: device.gpu_core_count(),
            max_threadgroup_memory,
            shared_memory_size: device.shared_memory_size(),
            supports_simd_group: device.supports_simd_group(),
            supports_simd_group_matrix: device.supports_simd_group_matrix(),
            supports_simd_reduction: device.supports_simd_reduction(),
            supports_simd_shuffle_and_fill: device.supports_simd_shuffle_and_fill(),
            supports_simd_shuffles_and_broadcast: device.supports_simd_shuffles_and_broadcast(),
            supports_mxu: device.supports_mxu(),
            supports_tls: device.supports_tls(),
        }
    }

    pub fn is_high_performance(&self) -> bool {
        self.tier.is_high_performance()
    }

    fn parse_architecture(architecture_name: &str) -> (DeviceGeneration, GpuTier) {
        let Some(suffix_start) = architecture_name.find("applegpu_g") else {
            return (DeviceGeneration::Unknown(0), GpuTier::Unknown('?'));
        };
        let after_prefix = &architecture_name[suffix_start + "applegpu_g".len()..];

        let digit_end = after_prefix
            .char_indices()
            .find(|(_, character)| !character.is_ascii_digit())
            .map_or(after_prefix.len(), |(index, _)| index);

        if digit_end == 0 {
            return (DeviceGeneration::Unknown(0), GpuTier::Unknown('?'));
        }

        let generation_number: u8 = match after_prefix[..digit_end].parse() {
            Ok(n) => n,
            Err(_) => return (DeviceGeneration::Unknown(0), GpuTier::Unknown('?')),
        };

        let tier = match after_prefix[digit_end..].chars().next() {
            Some('p') => GpuTier::Phone,
            Some('g') => GpuTier::Base,
            Some('s') => GpuTier::Max,
            Some('d') => GpuTier::Ultra,
            Some(other) => GpuTier::Unknown(other),
            None => GpuTier::Unknown('?'),
        };

        (DeviceGeneration::from_generation_number(generation_number), tier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_phone_architecture() {
        let (generation, tier) = MetalDeviceCapabilities::parse_architecture("applegpu_g13p");
        assert_eq!(generation, DeviceGeneration::Gen13);
        assert_eq!(tier, GpuTier::Phone);
    }

    #[test]
    fn parses_base_mac_architecture() {
        let (generation, tier) = MetalDeviceCapabilities::parse_architecture("applegpu_g14g");
        assert_eq!(generation, DeviceGeneration::Gen14);
        assert_eq!(tier, GpuTier::Base);
    }

    #[test]
    fn parses_max_mac_architecture() {
        let (generation, tier) = MetalDeviceCapabilities::parse_architecture("applegpu_g17s");
        assert_eq!(generation, DeviceGeneration::Gen17);
        assert_eq!(tier, GpuTier::Max);
    }

    #[test]
    fn parses_ultra_mac_architecture() {
        let (generation, tier) = MetalDeviceCapabilities::parse_architecture("applegpu_g18d");
        assert_eq!(generation, DeviceGeneration::Gen18);
        assert_eq!(tier, GpuTier::Ultra);
    }

    #[test]
    fn parses_m3_base_architecture() {
        let (generation, tier) = MetalDeviceCapabilities::parse_architecture("applegpu_g15g");
        assert_eq!(generation, DeviceGeneration::Gen15);
        assert_eq!(tier, GpuTier::Base);
    }

    #[test]
    fn parses_m3_pro_max_architecture() {
        let (generation, tier) = MetalDeviceCapabilities::parse_architecture("applegpu_g16s");
        assert_eq!(generation, DeviceGeneration::Gen16);
        assert_eq!(tier, GpuTier::Max);
    }

    #[test]
    fn returns_unknown_for_malformed_architecture() {
        let (generation, tier) = MetalDeviceCapabilities::parse_architecture("");
        assert_eq!(generation, DeviceGeneration::Unknown(0));
        assert_eq!(tier, GpuTier::Unknown('?'));

        let (generation, tier) = MetalDeviceCapabilities::parse_architecture("not_a_gpu");
        assert_eq!(generation, DeviceGeneration::Unknown(0));
        assert_eq!(tier, GpuTier::Unknown('?'));
    }

    #[test]
    fn returns_unknown_for_unrecognized_generation() {
        let (generation, tier) = MetalDeviceCapabilities::parse_architecture("applegpu_g99g");
        assert_eq!(generation, DeviceGeneration::Unknown(99));
        assert_eq!(tier, GpuTier::Base);
    }

    #[test]
    fn returns_unknown_tier_for_unrecognized_suffix() {
        let (generation, tier) = MetalDeviceCapabilities::parse_architecture("applegpu_g18z");
        assert_eq!(generation, DeviceGeneration::Gen18);
        assert_eq!(tier, GpuTier::Unknown('z'));
    }

    #[test]
    fn high_performance_only_for_max_and_ultra() {
        assert!(!GpuTier::Phone.is_high_performance());
        assert!(!GpuTier::Base.is_high_performance());
        assert!(GpuTier::Max.is_high_performance());
        assert!(GpuTier::Ultra.is_high_performance());
        assert!(!GpuTier::Unknown('x').is_high_performance());
    }
}
