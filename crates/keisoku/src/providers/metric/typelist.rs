use core::marker::PhantomData;
use std::any::{Any, TypeId};

pub struct Nil;
pub struct Cons<H, T>(PhantomData<(H, T)>);

pub struct Values<M: Metric, T> {
    head: M::Value,
    tail: T,
}

impl<M: Metric, T> Values<M, T> {
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

pub trait Metric: 'static {
    type Value: 'static;
    const TYPE_BIT: u128;
}

pub trait MetricSet {
    type Value: ValueList;
    const TYPE_MASK: u128;
}

impl MetricSet for Nil {
    type Value = Nil;
    const TYPE_MASK: u128 = 0;
}

impl<H, T> MetricSet for Cons<H, T>
where
    H: Metric,
    T: MetricSet,
{
    type Value = Values<H, T::Value>;
    const TYPE_MASK: u128 = {
        assert!(H::TYPE_BIT & T::TYPE_MASK == 0, "duplicate metric in Select!");
        H::TYPE_BIT | T::TYPE_MASK
    };
}

pub trait ValueList {
    fn get<T: Metric>(&self) -> Option<&T::Value>;
}

impl ValueList for Nil {
    fn get<T: Metric>(&self) -> Option<&T::Value> {
        None
    }
}

impl<H, Tail> ValueList for Values<H, Tail>
where
    H: Metric,
    Tail: ValueList,
{
    fn get<T: Metric>(&self) -> Option<&T::Value> {
        if TypeId::of::<H>() == TypeId::of::<T>() {
            return (&self.head as &dyn Any).downcast_ref::<T::Value>();
        }
        self.tail.get::<T>()
    }
}

/// Builds a recursive metric selector list.
///
/// Repeating the same metric marker in a provider selector is rejected at
/// compile time:
///
/// ```compile_fail
/// use keisoku::{Instant, Memory, Select};
///
/// let _ = Instant::<Select![Memory, Memory]>::new();
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
