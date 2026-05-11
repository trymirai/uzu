use debug_display::Display;
use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Debug, Display, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum QuantizationMethod {
    #[serde(rename = "mlx")]
    ScaleBias,
    #[serde(rename = "awq")]
    ScaleZeroPoint,
}
