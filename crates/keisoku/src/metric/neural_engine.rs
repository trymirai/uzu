use std::time::Duration;

use obfstr::obfstr;

use super::{groups::IoReportGroups, measured::Measured};
use crate::{
    decode::{self, GroupId, RawChannel},
    metrics::NeuralEngineMetrics,
    sources::Sources,
    units::Percent,
};

pub struct NeuralEngine;

impl Measured for NeuralEngine {
    type Value = NeuralEngineMetrics;
    type Ctx<'a> = ();
    type Acc = f32;
    const GROUPS: IoReportGroups = IoReportGroups::PMP;

    fn context(
        _sources: &Sources,
        _package_watts_mean: Option<f32>,
    ) {
    }

    fn consume(
        acc: &mut f32,
        channel: &RawChannel,
        _ctx: &(),
    ) {
        if channel.group == GroupId::Pmp
            && channel.subgroup.contains(obfstr!("Floor"))
            && (channel.name == obfstr!("ANE-AF-BW") || channel.name == obfstr!("ANE-DCS-BW"))
        {
            *acc = acc.max(decode::residency_active_percent(&channel.states));
        }
    }

    fn finish(
        acc: f32,
        _elapsed: Duration,
        _ctx: &(),
    ) -> NeuralEngineMetrics {
        NeuralEngineMetrics {
            active: Percent(acc),
        }
    }
}
