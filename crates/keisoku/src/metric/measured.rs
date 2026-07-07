use std::time::Duration;

use super::groups::IoReportGroups;
use crate::{decode::RawChannel, sources::Sources};

/// Interval metrics. Each declares exactly the IOReport channels it subscribes
/// to (`GROUPS`), the non-channel inputs it needs (`Ctx`), and how it folds the
/// window's channels into its value (`Acc` + `consume`/`finish`). Nothing shared
/// or unused is passed in — an energy-only interval never sees frequency tables.
pub trait Measured {
    type Value;
    /// Exactly this metric's non-channel inputs (frequency tables, package watts, …).
    type Ctx<'a>;
    /// Per-metric fold state, seeded before the single channel pass.
    type Acc: Default;
    const GROUPS: IoReportGroups;

    fn context(
        sources: &Sources,
        package_watts_mean: Option<f32>,
    ) -> Self::Ctx<'_>;
    fn consume(
        acc: &mut Self::Acc,
        channel: &RawChannel,
        ctx: &Self::Ctx<'_>,
    );
    fn finish(
        acc: Self::Acc,
        elapsed: Duration,
        ctx: &Self::Ctx<'_>,
    ) -> Self::Value;
}
