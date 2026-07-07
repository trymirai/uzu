use std::time::Duration;

use super::{groups::IoReportGroups, measured::Measured};
use crate::{
    decode::{EnergyTotals, GroupId, RawChannel},
    metrics::PowerMetrics,
    sources::Sources,
    units::Watts,
};

pub struct Power;

impl Measured for Power {
    type Value = PowerMetrics;
    type Ctx<'a> = Option<f32>;
    type Acc = EnergyTotals;
    const GROUPS: IoReportGroups = IoReportGroups::ENERGY_MODEL;

    fn context(
        _sources: &Sources,
        package_watts_mean: Option<f32>,
    ) -> Option<f32> {
        package_watts_mean
    }

    fn consume(
        acc: &mut EnergyTotals,
        channel: &RawChannel,
        _ctx: &Option<f32>,
    ) {
        if channel.group == GroupId::EnergyModel {
            acc.accumulate(&channel.name, channel.integer_value, &channel.unit);
        }
    }

    fn finish(
        acc: EnergyTotals,
        elapsed: Duration,
        ctx: &Option<f32>,
    ) -> PowerMetrics {
        let seconds = elapsed.as_secs_f64().max(0.001);
        let package = ctx.map(Watts).unwrap_or_else(|| Watts((acc.total() / seconds) as f32));
        PowerMetrics {
            cpu: Watts((acc.cpu / seconds) as f32),
            gpu: Watts((acc.gpu / seconds) as f32),
            ane: Watts((acc.ane / seconds) as f32),
            ram: Watts((acc.ram / seconds) as f32),
            package,
        }
    }
}
