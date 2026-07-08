use std::time::Duration;

use crate::sys::ioreport::decode::{AneActivity, CpuResidency, DramBandwidth, EnergyTotals, FrequencyTables, GpuResidency};

pub struct IntervalFrame<'a> {
    pub(crate) elapsed: Duration,
    pub(crate) energy: Option<EnergyTotals>,
    pub(crate) cpu: Option<CpuResidency>,
    pub(crate) gpu: Option<GpuResidency>,
    pub(crate) ane: Option<AneActivity>,
    pub(crate) bandwidth: Option<DramBandwidth>,
    pub(crate) frequencies: Option<FrequencyTables<'a>>,
}
