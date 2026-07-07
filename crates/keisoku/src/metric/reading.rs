use crate::sources::Sources;

/// Static + Instant metrics: every value meaningful from a single read.
pub trait Reading {
    type Value;
    fn read(sources: &mut Sources) -> Self::Value;
}
