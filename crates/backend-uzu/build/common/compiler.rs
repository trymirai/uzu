use std::collections::HashMap;

use async_trait::async_trait;
use igata::{enum_paths::EnumPaths, gpu_types::GpuTypes};

use super::{identifiers::KernelPath, kernel::Kernel};

#[async_trait]
pub trait Compiler {
    async fn build(
        &self,
        gpu_types: &GpuTypes,
        enum_paths: &EnumPaths,
    ) -> anyhow::Result<HashMap<KernelPath, Box<[Kernel]>>>;
}
