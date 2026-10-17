use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::common::{identifiers::KernelName, kernel::Kernel};

#[derive(Serialize, Deserialize)]
pub struct Dephashes {
    pub buildsystem_hash: [u8; blake3::OUT_LEN],
    pub dependency_hashes: HashMap<String, [u8; blake3::OUT_LEN]>,
    pub public_kernels: Box<[Kernel]>,
    /// Kernels bound without a common trait.
    pub private_kernels: Box<[KernelName]>,
}
