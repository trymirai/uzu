use core::marker::PhantomData;
use std::any::{Any, TypeId};

pub struct Nil;
pub struct Cons<H, T>(PhantomData<(H, T)>);

pub struct Values<M: ChannelMetric, T> {
    pub(crate) head: M::Value,
    pub(crate) tail: T,
}

impl<M: ChannelMetric, T> Values<M, T> {
    pub(crate) fn new(
        head: M::Value,
        tail: T,
    ) -> Self {
        Self {
            head,
            tail,
        }
    }
}

pub trait ChannelMetric: 'static {
    type Value: 'static;
    const TYPE_BIT: u128;
}

pub trait ChannelSet {
    type Value: ValueList;
    const TYPE_MASK: u128;
}

impl ChannelSet for Nil {
    type Value = Nil;
    const TYPE_MASK: u128 = 0;
}

impl<H, T> ChannelSet for Cons<H, T>
where
    H: ChannelMetric,
    T: ChannelSet,
{
    type Value = Values<H, T::Value>;
    const TYPE_MASK: u128 = {
        assert!(H::TYPE_BIT & T::TYPE_MASK == 0, "duplicate channel in Select!");
        H::TYPE_BIT | T::TYPE_MASK
    };
}

pub trait ValueList {
    fn get<T: ChannelMetric>(&self) -> Option<&T::Value>;
}

impl ValueList for Nil {
    fn get<T: ChannelMetric>(&self) -> Option<&T::Value> {
        None
    }
}

impl<H, Tail> ValueList for Values<H, Tail>
where
    H: ChannelMetric,
    Tail: ValueList,
{
    fn get<T: ChannelMetric>(&self) -> Option<&T::Value> {
        if TypeId::of::<H>() == TypeId::of::<T>() {
            return (&self.head as &dyn Any).downcast_ref::<T::Value>();
        }
        self.tail.get::<T>()
    }
}

/// Builds a compile-time list of IOReport channel markers for [`Device::interval_measurement`](crate::Device::interval_measurement).
///
/// # Examples
///
/// ```ignore
/// use keisoku::{Cpu, Device, EnergyRail, Select};
///
/// let mut handle = Device::interval_measurement::<Select![EnergyRail<Cpu>]>();
/// ```
#[macro_export]
macro_rules! Select {
    () => {
        $crate::Nil
    };
    ($head:ty $(, $rest:ty)* $(,)?) => {
        $crate::Cons<$head, $crate::Select![$($rest),*]>
    };
}
