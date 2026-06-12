use rand::prelude::*;

use crate::session::parameter::ResolvableValue;

#[derive(Debug, Clone, Copy, Default)]
pub enum SamplingSeed {
    #[default]
    Default,
    Custom(u64),
}

impl ResolvableValue<u64> for SamplingSeed {
    fn resolve(&self) -> u64 {
        match self {
            SamplingSeed::Default => rand::rng().random::<u64>(),
            SamplingSeed::Custom(seed) => *seed,
        }
    }
}
