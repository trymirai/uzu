use std::collections::HashMap;

use async_trait::async_trait;

use super::kernel::{Kernel, Struct};

pub struct BuildResult {
    pub kernels: HashMap<Box<[Box<str>]>, Box<[Kernel]>>,
    pub structs: Vec<Struct>,
}

#[async_trait]
pub trait Compiler {
    async fn build(&self) -> anyhow::Result<BuildResult>;
}
