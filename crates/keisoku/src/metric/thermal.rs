use super::reading::Reading;
use crate::{metrics::ThermalPressure, sources::Sources};

pub struct Thermal;

impl Reading for Thermal {
    type Value = Option<ThermalPressure>;

    fn read(_sources: &mut Sources) -> Option<ThermalPressure> {
        ThermalPressure::read()
    }
}
