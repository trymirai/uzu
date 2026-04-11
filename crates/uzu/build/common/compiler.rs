use std::collections::HashMap;

use async_trait::async_trait;

use super::{gpu_types::GpuTypes, kernel::Kernel};

#[async_trait]
pub trait Compiler {
    async fn build(
        &self,
        gpu_types: &GpuTypes,
    ) -> anyhow::Result<HashMap<Box<[Box<str>]>, Box<[Kernel]>>>;
}
