use std::{collections::HashMap, fs};

use anyhow::Context;
use serde::{Deserialize, Serialize};

pub use super::cache_protocol::{SharedArtifactCache, air_key, dependencies_match, index_key, metallib_key, zstd_key};
use super::{ast::MetalKernelInfo, cache_protocol::Stage};
use crate::common::{caching, identifiers::KernelName, kernel::Kernel};

#[derive(Serialize, Deserialize, Clone)]
pub struct SourceIndex {
    pub schema: u32,
    pub source_hash: [u8; blake3::OUT_LEN],
    pub compile_schema_hash: [u8; blake3::OUT_LEN],
    pub analyzer_hash: [u8; blake3::OUT_LEN],
    pub gpu_types_hash: [u8; blake3::OUT_LEN],
    pub dependency_hashes: HashMap<Box<str>, [u8; blake3::OUT_LEN]>,
    pub footer: String,
    pub kernel_infos: Vec<MetalKernelInfo>,
    pub specialize_indices: HashMap<KernelName, usize>,
    pub public_kernels: Box<[Kernel]>,
}

pub fn load_source_index(
    cache: &SharedArtifactCache,
    key: &blake3::Hash,
) -> anyhow::Result<Option<SourceIndex>> {
    let Some(path) = cache.lookup(Stage::Index, key)? else {
        return Ok(None);
    };
    let bytes = fs::read(&path.path).with_context(|| format!("cannot read {}", path.path.display()))?;
    let Ok(index) = serde_json::from_slice::<SourceIndex>(&bytes) else {
        return Ok(None);
    };
    if index.schema != caching::METAL_CACHE_SCHEMA {
        return Ok(None);
    }
    Ok(Some(index))
}

pub fn store_source_index(
    cache: &SharedArtifactCache,
    key: &blake3::Hash,
    index: &SourceIndex,
) -> anyhow::Result<()> {
    let bytes = serde_json::to_vec_pretty(index).context("cannot serialize source index")?;
    cache.store_bytes(Stage::Index, key, bytes)?;
    Ok(())
}
