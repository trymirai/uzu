use super::{
    Cons as ConsType, Nil as NilType,
    typelist::{MetricSet, Values},
};
use crate::sources::Sources;

pub trait InstantMetric: super::typelist::Metric {
    fn read(sources: &mut Sources) -> Self::Value;
}

pub trait InstantSet: MetricSet {
    fn read(sources: &mut Sources) -> Self::Value;
}

impl InstantSet for NilType {
    fn read(_sources: &mut Sources) -> Self::Value {
        NilType
    }
}

impl<H, T> InstantSet for ConsType<H, T>
where
    H: InstantMetric,
    T: InstantSet,
{
    fn read(sources: &mut Sources) -> Self::Value {
        Values::new(H::read(sources), T::read(sources))
    }
}
