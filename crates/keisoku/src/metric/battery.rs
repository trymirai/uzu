use super::reading::Reading;
use crate::{metrics::BatteryMetrics, sources::Sources};

pub struct Battery;

impl Reading for Battery {
    type Value = Option<BatteryMetrics>;

    fn read(_sources: &mut Sources) -> Option<BatteryMetrics> {
        BatteryMetrics::read()
    }
}
