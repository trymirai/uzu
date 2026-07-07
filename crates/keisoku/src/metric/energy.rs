use std::time::Duration;

use super::{groups::IoReportGroups, measured::Measured};
use crate::{
    decode::{EnergyTotals, GroupId, RawChannel},
    metrics::EnergyMetrics,
    sources::Sources,
    units::Joules,
};

pub struct Energy;

impl Measured for Energy {
    type Value = EnergyMetrics;
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
    ) -> EnergyMetrics {
        let seconds = elapsed.as_secs_f32().max(0.001);
        let package = ctx.map(|watts| Joules(watts * seconds)).unwrap_or(Joules(acc.total() as f32));
        EnergyMetrics {
            cpu: Joules(acc.cpu as f32),
            gpu: Joules(acc.gpu as f32),
            ane: Joules(acc.ane as f32),
            ram: Joules(acc.ram as f32),
            package,
        }
    }
}
