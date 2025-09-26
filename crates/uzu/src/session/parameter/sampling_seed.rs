use rand::prelude::*;

use crate::session::parameter::ResolvableValue;

#[derive(Debug, Clone, Copy)]
pub enum SamplingSeed {
    Default,
    Custom(u64),
}

impl Default for SamplingSeed {
    fn default() -> Self {
        SamplingSeed::Default
    }
}

impl ResolvableValue<u64> for SamplingSeed {
    fn resolve(&self) -> u64 {
        match self {
            SamplingSeed::Default => rand::rng().random::<u64>(),
            SamplingSeed::Custom(seed) => *seed,
        }
    }
}
