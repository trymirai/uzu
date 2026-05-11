use std::collections::HashMap;

use async_trait::async_trait;

use super::{enum_paths::EnumPaths, gpu_types::GpuTypes, kernel::Kernel};

#[async_trait]
pub trait Compiler {
    async fn build(
        &self,
        gpu_types: &GpuTypes,
        enum_paths: &EnumPaths,
    ) -> anyhow::Result<HashMap<Box<[Box<str>]>, Box<[Kernel]>>>;
}
