use core::marker::PhantomData;

use super::typelist::{ChannelMetric, ChannelSet, ValueList};

/// Typed IOReport channel values selected by [`Select!`](crate::Select).
#[must_use]
pub struct Sample<M: ChannelSet> {
    values: M::Value,
    marker: PhantomData<M>,
}

impl<M: ChannelSet> Sample<M> {
    pub(crate) fn new(values: M::Value) -> Self {
        Self {
            values,
            marker: PhantomData,
        }
    }

    /// Returns the value for channel marker `T`.
    ///
    /// # Panics
    ///
    /// Panics if `T` was not included in the `Select!` list used to create this sample.
    pub fn get<T: ChannelMetric>(&self) -> &T::Value {
        self.values.get::<T>().expect("channel not in sample")
    }
}
