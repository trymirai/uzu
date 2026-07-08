use super::{
    Cons as ConsType, Nil as NilType,
    typelist::{MetricSet, Values},
};
use crate::sources::interval::{IntervalFrame, IntervalInputs};

pub trait IntervalMetric: super::typelist::Metric {
    const INPUTS: IntervalInputs;
    fn finish(frame: &IntervalFrame<'_>) -> Self::Value;
}

pub trait IntervalSet: MetricSet {
    const INPUTS: IntervalInputs;
    fn finish(frame: &IntervalFrame<'_>) -> Self::Value;
}

impl IntervalSet for NilType {
    const INPUTS: IntervalInputs = IntervalInputs::empty();

    fn finish(_frame: &IntervalFrame<'_>) -> Self::Value {
        NilType
    }
}

impl<H, T> IntervalSet for ConsType<H, T>
where
    H: IntervalMetric,
    T: IntervalSet,
{
    const INPUTS: IntervalInputs = IntervalInputs::from_bits_retain(H::INPUTS.bits() | T::INPUTS.bits());

    fn finish(frame: &IntervalFrame<'_>) -> Self::Value {
        Values::new(H::finish(frame), T::finish(frame))
    }
}
