use core::marker::PhantomData;

use crate::{
    providers::metric::{InstantSet, Sample},
    sources::Sources,
};

pub struct Instant<M: InstantSet> {
    sources: Sources,
    marker: PhantomData<M>,
}

impl<M: InstantSet> Instant<M> {
    pub fn new() -> Self {
        let _ = M::TYPE_MASK;
        Self {
            sources: Sources::new(),
            marker: PhantomData,
        }
    }

    pub fn read(&mut self) -> Sample<M> {
        Sample::new(M::read(&mut self.sources))
    }
}

impl<M: InstantSet> Default for Instant<M> {
    fn default() -> Self {
        Self::new()
    }
}
