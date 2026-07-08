use std::time::Duration;

use crate::{
    providers::marker::{CpuResidency, DramBandwidth},
    sys::ioreport::decode::{EnergyTotals, FrequencyTables},
};

pub struct IntervalFrame<'a> {
    pub(crate) elapsed: Duration,
    pub(crate) energy: Option<EnergyTotals>,
    pub(crate) package_watts_mean: Option<f32>,
    pub(crate) cpu: Option<CpuResidency>,
    pub(crate) gpu: Option<(u32, f32)>,
    pub(crate) ane: Option<f32>,
    pub(crate) bandwidth: Option<DramBandwidth>,
    pub(crate) frequencies: Option<FrequencyTables<'a>>,
}
