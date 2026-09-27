use std::collections::HashMap;

use super::{enum_paths::EnumPaths, gpu_types::GpuTypes, identifiers::KernelPath, kernel::Kernel};

pub trait Compiler {
    fn build(
        &self,
        gpu_types: &GpuTypes,
        enum_paths: &EnumPaths,
    ) -> anyhow::Result<HashMap<KernelPath, Box<[Kernel]>>>;
}
