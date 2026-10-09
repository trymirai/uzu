use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::common::kernel::Kernel;

#[derive(Serialize, Deserialize)]
pub struct Cached {
    pub cache_key: [u8; blake3::OUT_LEN],
    pub dependency_hashes: HashMap<Box<str>, [u8; blake3::OUT_LEN]>,
    pub public_kernels: Box<[Kernel]>,
    pub has_kernels: bool,
    pub num_variants: usize,
    pub num_shards: usize,
}
