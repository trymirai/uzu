use crate::common::{gpu_type_paths::GpuTypePaths, gpu_types::GpuTypes};

pub struct SlangGpuTypes<'a> {
    pub paths: GpuTypePaths,
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    pub definitions: Vec<&'a GpuTypes>,
}
