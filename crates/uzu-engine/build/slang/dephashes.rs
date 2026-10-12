use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::common::{identifiers::KernelName, kernel::Kernel};

#[derive(Serialize, Deserialize)]
pub struct Dephashes {
    pub buildsystem_hash: [u8; blake3::OUT_LEN],
    pub artifact_hashes: HashMap<String, [u8; blake3::OUT_LEN]>,
    pub dependency_hashes: HashMap<String, [u8; blake3::OUT_LEN]>,
    pub public_kernels: Box<[Kernel]>,
    /// Every non-public kernel's binding: `true` for a `[[Test]]` kernel, bound in test builds only, `false` for a
    /// production-private one, bound in every build without a common trait.
    pub bindings: Box<[(KernelName, bool)]>,
}
