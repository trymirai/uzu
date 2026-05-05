use debug_display::Display;
use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Debug, Display, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum QuantizationMethod {
    #[serde(rename = "mlx")]
    MLX,
    #[serde(rename = "awq")]
    AWQ,
}
