use std::collections::HashMap;

use crate::common::{gpu_type_paths::GpuTypePaths, gpu_types::GpuTypes};

/// The GPU types a target's kernels name: the common ones and the target's own.
pub struct SlangGpuTypes<'a> {
    /// The canonical path of each named type by its short name.
    pub map: HashMap<String, String>,
    pub paths: GpuTypePaths,
    // Read by the Vulkan binding's layout and precondition checks.
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    pub definitions: Vec<&'a GpuTypes>,
}
