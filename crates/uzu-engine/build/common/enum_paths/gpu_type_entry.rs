#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "vulkan"))]
use super::GpuTypeKind;
use crate::common::gpu_types::GpuTypePath;

#[derive(Clone)]
pub struct GpuTypeEntry {
    pub path: GpuTypePath,
    /// `None` for a struct, which lowers to no scalar.
    #[cfg(any(all(feature = "metal", target_os = "macos"), feature = "vulkan"))]
    pub kind: Option<GpuTypeKind>,
}
