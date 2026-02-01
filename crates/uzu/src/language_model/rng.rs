pub struct PRng {
    seed: u64,
}

impl PRng {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
        }
    }

    pub fn derive(
        &self,
        index: u64,
    ) -> u64 {
        let mut hash = self.seed.wrapping_add(index);
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xff51afd7ed558ccd);
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xc4ceb9fe1a85ec53);
        hash ^= hash >> 33;
        hash
    }
}
