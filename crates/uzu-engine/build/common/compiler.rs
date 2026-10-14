use std::collections::HashMap;

use super::{gpu_type_paths::GpuTypePaths, gpu_types::GpuTypes, identifiers::KernelPath, kernel::Kernel};

pub trait Compiler {
    fn build(
        &self,
        gpu_types: &GpuTypes,
        gpu_type_paths: &GpuTypePaths,
    ) -> anyhow::Result<HashMap<KernelPath, Box<[Kernel]>>>;
}
