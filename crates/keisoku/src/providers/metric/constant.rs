use super::{
    Cons as ConsType, Nil as NilType,
    typelist::{MetricSet, Values},
};
use crate::sources::Sources;

pub trait ConstantMetric: super::typelist::Metric {
    fn read(sources: &mut Sources) -> Self::Value;
}

pub trait ConstantSet: MetricSet {
    fn read(sources: &mut Sources) -> Self::Value;
}

impl ConstantSet for NilType {
    fn read(_sources: &mut Sources) -> Self::Value {
        NilType
    }
}

impl<H, T> ConstantSet for ConsType<H, T>
where
    H: ConstantMetric,
    T: ConstantSet,
{
    fn read(sources: &mut Sources) -> Self::Value {
        Values::new(H::read(sources), T::read(sources))
    }
}
