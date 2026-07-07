use core::marker::PhantomData;

use crate::{metric::Reading, sources::Sources};

pub struct Instant<M: Reading> {
    sources: Sources,
    marker: PhantomData<M>,
}

impl<M: Reading> Instant<M> {
    pub fn new() -> Self {
        Self {
            sources: Sources::new(),
            marker: PhantomData,
        }
    }

    pub fn read(&mut self) -> M::Value {
        M::read(&mut self.sources)
    }
}

impl<M: Reading> Default for Instant<M> {
    fn default() -> Self {
        Self::new()
    }
}
