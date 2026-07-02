use crate::{PowerMetrics, metrics::EnergyMetrics, units::Milliseconds};

#[derive(Debug, Clone)]
pub struct EnergyReading {
    pub energy: EnergyMetrics,
    pub average_power: PowerMetrics,
    pub elapsed: Milliseconds,
}
