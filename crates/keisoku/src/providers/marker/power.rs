use super::{interval_set::IntervalMetric, typelist::Metric};
use crate::{
    providers::data::PowerMetrics,
    sources::interval::{IntervalFrame, IntervalInputs},
    units::Watts,
};

pub struct Power;

impl Metric for Power {
    type Value = PowerMetrics;
    const TYPE_BIT: u128 = 1 << 16;
}

impl IntervalMetric for Power {
    const INPUTS: IntervalInputs = IntervalInputs::ENERGY_RAILS;

    fn finish(frame: &IntervalFrame<'_>) -> PowerMetrics {
        let acc = frame.energy.as_ref().cloned().unwrap_or_default();
        let seconds = frame.elapsed.as_secs_f64().max(0.001);
        PowerMetrics {
            cpu: Watts((acc.cpu / seconds) as f32),
            gpu: Watts((acc.gpu / seconds) as f32),
            ane: Watts((acc.ane / seconds) as f32),
            ram: Watts((acc.ram / seconds) as f32),
        }
    }
}
