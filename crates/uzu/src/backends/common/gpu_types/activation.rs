use num_traits::Float;
use serde::{Deserialize, Serialize};

const GELU_K0: f32 = 0.044715;
const GELU_K1: f32 = 0.7978845608; // sqrt(2/pi)

#[repr(C)]
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ActivationType {
    #[serde(rename = "SiLU")]
    SILU {
        #[serde(default = "default_silu_alpha")]
        alpha: f32,
    },
    #[serde(rename = "GELU")]
    GELU,
    #[serde(rename = "Identity")]
    IDENTITY,
}

impl ActivationType {
    pub fn silu_default() -> ActivationType {
        ActivationType::SILU {
            alpha: 1.0,
        }
    }

    pub fn activate<T: Float>(
        &self,
        x: T,
    ) -> T {
        match self {
            ActivationType::SILU {
                ..
            } => silu(x),
            ActivationType::GELU => gelu(x),
            ActivationType::IDENTITY => x,
        }
    }

    pub fn alpha(&self) -> f32 {
        match self {
            ActivationType::SILU {
                alpha,
            } => *alpha,
            ActivationType::GELU => 1.0,
            ActivationType::IDENTITY => 1.0,
        }
    }
}

fn silu<T: Float>(x: T) -> T {
    let x_float = x.to_f32().unwrap();
    let y_float = x_float / (1.0f32 + (-x_float).exp());
    T::from(y_float).unwrap()
}

fn gelu<T: Float>(x: T) -> T {
    let x_float = x.to_f32().unwrap();
    let tan_arg = GELU_K1 * (x_float + GELU_K0 * x_float * x_float * x_float);
    let y_float = 0.5f32 * x_float * (1.0f32 + tan_arg.tanh());
    T::from(y_float).unwrap()
}

fn default_silu_alpha() -> f32 {
    1.0
}
