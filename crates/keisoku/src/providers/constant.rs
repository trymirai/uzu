use crate::{
    providers::marker::{ConstantSet, Sample},
    sources::Sources,
};

pub struct Constant<M: ConstantSet> {
    sample: Sample<M>,
}

impl<M: ConstantSet> Constant<M> {
    pub fn new() -> Self {
        let _ = M::TYPE_MASK;
        let mut sources = Sources::new();
        Self {
            sample: Sample::new(M::read(&mut sources)),
        }
    }

    pub fn sample(&self) -> &Sample<M> {
        &self.sample
    }

    pub fn into_sample(self) -> Sample<M> {
        self.sample
    }
}

impl<M: ConstantSet> Default for Constant<M> {
    fn default() -> Self {
        Self::new()
    }
}
