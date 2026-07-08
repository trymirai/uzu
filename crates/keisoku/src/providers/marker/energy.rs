use super::{interval_set::IntervalMetric, typelist::Metric};
use crate::{
    providers::data::EnergyMetrics,
    sources::interval::{IntervalFrame, IntervalInputs},
    units::Joules,
};

pub struct Energy;

impl Metric for Energy {
    type Value = EnergyMetrics;
    const TYPE_BIT: u128 = 1 << 15;
}

impl IntervalMetric for Energy {
    const INPUTS: IntervalInputs = IntervalInputs::ENERGY_RAILS;

    fn finish(frame: &IntervalFrame<'_>) -> EnergyMetrics {
        let acc = frame.energy.as_ref().cloned().unwrap_or_default();
        EnergyMetrics {
            cpu: Joules(acc.cpu as f32),
            gpu: Joules(acc.gpu as f32),
            ane: Joules(acc.ane as f32),
            ram: Joules(acc.ram as f32),
        }
    }
}
