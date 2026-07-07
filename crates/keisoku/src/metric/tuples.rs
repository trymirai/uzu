use std::time::Duration;

use super::{groups::IoReportGroups, measured::Measured, reading::Reading};
use crate::{decode::RawChannel, sources::Sources};

macro_rules! tuple_impls {
    ($($member:ident $index:tt),+) => {
        impl<$($member: Reading),+> Reading for ($($member,)+) {
            type Value = ($($member::Value,)+);

            fn read(sources: &mut Sources) -> Self::Value {
                ($($member::read(sources),)+)
            }
        }

        impl<$($member: Measured),+> Measured for ($($member,)+) {
            type Value = ($($member::Value,)+);
            type Ctx<'a> = ($($member::Ctx<'a>,)+);
            type Acc = ($($member::Acc,)+);
            const GROUPS: IoReportGroups = IoReportGroups::empty()$(.union($member::GROUPS))+;

            fn context(sources: &Sources, package_watts_mean: Option<f32>) -> Self::Ctx<'_> {
                ($($member::context(sources, package_watts_mean),)+)
            }

            fn consume(acc: &mut Self::Acc, channel: &RawChannel, ctx: &Self::Ctx<'_>) {
                $($member::consume(&mut acc.$index, channel, &ctx.$index);)+
            }

            fn finish(acc: Self::Acc, elapsed: Duration, ctx: &Self::Ctx<'_>) -> Self::Value {
                ($($member::finish(acc.$index, elapsed, &ctx.$index),)+)
            }
        }
    };
}

tuple_impls!(A 0);
tuple_impls!(A 0, B 1);
tuple_impls!(A 0, B 1, C 2);
tuple_impls!(A 0, B 1, C 2, D 3);
tuple_impls!(A 0, B 1, C 2, D 3, E 4);
tuple_impls!(A 0, B 1, C 2, D 3, E 4, F 5);
tuple_impls!(A 0, B 1, C 2, D 3, E 4, F 5, G 6);
tuple_impls!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);
tuple_impls!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8);
tuple_impls!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9);
tuple_impls!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10);
tuple_impls!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11);

impl Reading for () {
    type Value = ();

    fn read(_sources: &mut Sources) {}
}

impl Measured for () {
    type Value = ();
    type Ctx<'a> = ();
    type Acc = ();
    const GROUPS: IoReportGroups = IoReportGroups::empty();

    fn context(
        _sources: &Sources,
        _package_watts_mean: Option<f32>,
    ) {
    }

    fn consume(
        _acc: &mut (),
        _channel: &RawChannel,
        _ctx: &(),
    ) {
    }

    fn finish(
        _acc: (),
        _elapsed: Duration,
        _ctx: &(),
    ) {
    }
}
