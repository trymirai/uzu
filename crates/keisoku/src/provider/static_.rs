use crate::{metric::Reading, sources::Sources};

pub struct Static<M: Reading> {
    value: M::Value,
}

impl<M: Reading> Static<M> {
    pub fn new() -> Self {
        let mut sources = Sources::new();
        Self {
            value: M::read(&mut sources),
        }
    }

    pub fn get(&self) -> &M::Value {
        &self.value
    }

    pub fn into_inner(self) -> M::Value {
        self.value
    }
}

impl<M: Reading> Default for Static<M> {
    fn default() -> Self {
        Self::new()
    }
}
