use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::common::kernel::Kernel;

#[derive(Serialize, Deserialize)]
pub struct Cached {
    pub buildsystem_hash: [u8; blake3::OUT_LEN],
    pub compiler_version: String,
    pub dependency_hashes: HashMap<String, [u8; blake3::OUT_LEN]>,
    pub public_kernels: Box<[Kernel]>,
}
