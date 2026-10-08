use super::GpuTypeKind;
use crate::common::gpu_types::GpuTypePath;

#[derive(Clone)]
pub struct GpuTypeEntry {
    pub path: GpuTypePath,
    /// `None` for a struct, which lowers to no scalar.
    pub kind: Option<GpuTypeKind>,
}
