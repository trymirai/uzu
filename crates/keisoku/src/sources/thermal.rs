use crate::metrics::ThermalPressure;

pub(crate) fn read_thermal() -> Option<ThermalPressure> {
    let level = crate::sys::read_thermal_pressure()?;
    Some(match level {
        crate::sys::ThermalPressureLevel::Nominal => ThermalPressure::Nominal,
        crate::sys::ThermalPressureLevel::Moderate => ThermalPressure::Moderate,
        crate::sys::ThermalPressureLevel::Heavy => ThermalPressure::Heavy,
        crate::sys::ThermalPressureLevel::Trapping => ThermalPressure::Trapping,
        crate::sys::ThermalPressureLevel::Sleeping => ThermalPressure::Sleeping,
    })
}
