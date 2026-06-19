use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Temperatures {
    pub cpu_average: f32,
    pub gpu_average: f32,
}
