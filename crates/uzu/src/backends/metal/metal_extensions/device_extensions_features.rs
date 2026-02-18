use std::collections::HashSet;

use metal::{MTLDevice, MTLDeviceExt, MTLFeatureSet, MTLGPUFamily, MTLPixelFormat, MTLReadWriteTextureTier};
use objc2::runtime::ProtocolObject;

/// Enum representing various Metal features that may or may not be supported by a device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Feature {
    /// Non-uniform threadgroups allow for more flexible compute kernel dispatches.
    NonUniformThreadgroups,
    /// Tile shaders enable efficient rendering of tiled resources.
    TileShaders,
    /// Read-write textures allow both reading from and writing to textures in a shader.
    ReadWriteTextures(MTLPixelFormat),
    /// Ability to read and write to cube map textures in Metal functions.
    ReadWriteCubeMapTexturesInFunctions,
}

/// Extensions for metal::Device to provide feature checking capabilities
pub trait DeviceFeatures {
    /// Checks if the device supports a specific Metal feature.
    ///
    /// - Parameter feature: The feature to check for support.
    /// - Returns: A boolean indicating whether the feature is supported.
    fn supports_feature(
        &self,
        feature: Feature,
    ) -> bool;
}

impl DeviceFeatures for ProtocolObject<dyn MTLDevice> {
    fn supports_feature(
        &self,
        feature: Feature,
    ) -> bool {
        match feature {
            Feature::NonUniformThreadgroups => {
                #[allow(deprecated)]
                #[cfg(target_os = "ios")]
                {
                    self.supports_feature_set(MTLFeatureSet::iOS_GPUFamily4_v1)
                }
                #[cfg(target_os = "macos")]
                {
                    self.supports_feature_set(MTLFeatureSet::macOS_GPUFamily1_v3)
                }
                #[cfg(not(any(target_os = "ios", target_os = "macos")))]
                {
                    false
                }
            },
            Feature::TileShaders => self.supports_family(MTLGPUFamily::Apple4),
            Feature::ReadWriteTextures(pixel_format) => {
                let tier_one_supported_formats: HashSet<MTLPixelFormat> =
                    [MTLPixelFormat::R32Float, MTLPixelFormat::R32Uint, MTLPixelFormat::R32Sint].into();

                let tier_two_supported_formats: HashSet<MTLPixelFormat> = tier_one_supported_formats
                    .union(
                        &[
                            MTLPixelFormat::RGBA32Float,
                            MTLPixelFormat::RGBA32Uint,
                            MTLPixelFormat::RGBA32Sint,
                            MTLPixelFormat::RGBA16Float,
                            MTLPixelFormat::RGBA16Uint,
                            MTLPixelFormat::RGBA16Sint,
                            MTLPixelFormat::RGBA8Unorm,
                            MTLPixelFormat::RGBA8Uint,
                            MTLPixelFormat::RGBA8Sint,
                            MTLPixelFormat::R16Float,
                            MTLPixelFormat::R16Uint,
                            MTLPixelFormat::R16Sint,
                            MTLPixelFormat::R8Unorm,
                            MTLPixelFormat::R8Uint,
                            MTLPixelFormat::R8Sint,
                        ]
                        .into_iter()
                        .collect(),
                    )
                    .cloned()
                    .collect();

                match self.read_write_texture_support() {
                    MTLReadWriteTextureTier::Tier1 => tier_one_supported_formats.contains(&pixel_format),
                    MTLReadWriteTextureTier::Tier2 => tier_two_supported_formats.contains(&pixel_format),
                    MTLReadWriteTextureTier::None => false,
                }
            },
            Feature::ReadWriteCubeMapTexturesInFunctions => {
                let families_with_read_write_cube_map_support = [
                    MTLGPUFamily::Apple4,
                    MTLGPUFamily::Apple5,
                    MTLGPUFamily::Apple6,
                    MTLGPUFamily::Apple7,
                    MTLGPUFamily::Apple8,
                    MTLGPUFamily::Mac2,
                    // Note: Metal3 family is not included since we'd need to check for availability
                    // and conditional compilation
                ];

                for family in families_with_read_write_cube_map_support {
                    if self.supports_family(family) {
                        return true;
                    }
                }
                false
            },
        }
    }
}
