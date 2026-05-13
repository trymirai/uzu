use std::fmt;

use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum QuantizationMethod {
    #[serde(rename = "mlx")]
    ScaleBias,
    #[serde(rename = "awq")]
    ScaleZeroPoint,
}

impl fmt::Display for QuantizationMethod {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}
