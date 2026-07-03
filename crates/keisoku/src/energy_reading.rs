use crate::{PowerMetrics, metrics::EnergyMetrics, units::Milliseconds};

#[derive(Debug, Clone)]
pub struct EnergyReading {
    pub energy: EnergyMetrics,
    pub average_power: PowerMetrics,
    pub elapsed: Milliseconds,
    pub package_from_smc: bool,
}
