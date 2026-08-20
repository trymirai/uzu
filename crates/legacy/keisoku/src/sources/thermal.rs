use crate::metrics::ThermalPressure;

pub(crate) fn read_thermal() -> Option<ThermalPressure> {
    crate::sys::read_thermal_pressure()
}
