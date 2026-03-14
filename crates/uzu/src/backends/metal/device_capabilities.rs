use byte_unit::Byte;
use metal::{MTLDevice, MTLDeviceExt as _};
use objc2::runtime::ProtocolObject;

use super::metal_extensions::{DeviceExt, DeviceGeneration};
use crate::backends::common::DeviceCapabilities;

const HIGH_PERFORMANCE_CORE_THRESHOLD: u32 = 10;

#[derive(Debug, Clone)]
pub struct MetalDeviceCapabilities {
    pub generation: DeviceGeneration,
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
        let generation = Self::parse_generation(&architecture_name);
        let max_threadgroup_memory = Byte::from_u64(device.max_threadgroup_memory_length() as u64);

        Self {
            generation,
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
        self.gpu_core_count > HIGH_PERFORMANCE_CORE_THRESHOLD
    }

    fn parse_generation(architecture_name: &str) -> DeviceGeneration {
        let Some(suffix_start) = architecture_name.find("applegpu_g") else {
            return DeviceGeneration::Unknown(0);
        };
        let after_prefix = &architecture_name[suffix_start + "applegpu_g".len()..];

        let digit_end = after_prefix
            .char_indices()
            .find(|(_, character)| !character.is_ascii_digit())
            .map_or(after_prefix.len(), |(index, _)| index);

        if digit_end == 0 {
            return DeviceGeneration::Unknown(0);
        }

        match after_prefix[..digit_end].parse::<u8>() {
            Ok(n) => DeviceGeneration::from_generation_number(n),
            Err(_) => DeviceGeneration::Unknown(0),
        }
    }
}