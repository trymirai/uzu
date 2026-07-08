use core::marker::PhantomData;

use super::typelist::{Metric, MetricSet, ValueList};

pub struct Sample<M: MetricSet> {
    values: M::Value,
    marker: PhantomData<M>,
}

impl<M: MetricSet> Sample<M> {
    pub(crate) fn new(values: M::Value) -> Self {
        Self {
            values,
            marker: PhantomData,
        }
    }

    pub fn get<T: Metric>(&self) -> &T::Value {
        self.try_get::<T>().expect("metric not in sample")
    }

    pub fn try_get<T: Metric>(&self) -> Option<&T::Value> {
        self.values.get::<T>()
    }
}
