use serde::{Deserialize, Serialize};

use crate::units::Celsius;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Temperatures {
    pub cpu_average: Option<Celsius>,
    pub gpu_average: Option<Celsius>,
}
