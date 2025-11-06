pub struct DerivableSeed {
    base_seed: u64,
    invocation: u64,
}

impl DerivableSeed {
    pub fn new(seed: u64) -> Self {
        Self {
            base_seed: seed,
            invocation: 0,
        }
    }

    pub fn current(&self) -> u64 {
        let mut hash = self.base_seed.wrapping_add(self.invocation);
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xff51afd7ed558ccd);
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xc4ceb9fe1a85ec53);
        hash ^= hash >> 33;
        hash
    }

    pub fn next(&mut self) -> u64 {
        self.invocation += 1;

        self.current()
    }
}
