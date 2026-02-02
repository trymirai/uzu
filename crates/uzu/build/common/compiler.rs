use std::collections::HashMap;

use async_trait::async_trait;

use super::kernel::Kernel;

#[async_trait]
pub trait Compiler {
    async fn build(
        &self
    ) -> anyhow::Result<HashMap<Box<[Box<str>]>, Box<[Kernel]>>>;
}
