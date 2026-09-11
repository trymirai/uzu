use crate::metrics::ThermalPressure;

pub fn read_thermal() -> Option<ThermalPressure> {
    crate::sys::read_thermal_pressure()
}
