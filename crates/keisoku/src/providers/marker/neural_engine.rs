use super::{interval_set::IntervalMetric, typelist::Metric};
use crate::{
    providers::data::NeuralEngineMetrics,
    sources::interval::{IntervalFrame, IntervalInputs},
    units::Percent,
};

pub struct NeuralEngine;

impl Metric for NeuralEngine {
    type Value = NeuralEngineMetrics;
    const TYPE_BIT: u128 = 1 << 19;
}

impl IntervalMetric for NeuralEngine {
    const INPUTS: IntervalInputs = IntervalInputs::ANE_ACTIVITY;

    fn finish(frame: &IntervalFrame<'_>) -> NeuralEngineMetrics {
        NeuralEngineMetrics {
            active: Percent(frame.ane.unwrap_or(0.0)),
        }
    }
}
