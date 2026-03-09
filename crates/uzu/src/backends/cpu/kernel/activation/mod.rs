use num_traits::Float;

use crate::ArrayElement;

pub mod activation;

const GELU_K0: f32 = 0.044715f32;
const GELU_K1: f32 = 0.7978845608; // sqrt(2/pi)

#[repr(u32)]
pub enum ActivationType {
    SiLU,
    GELU,
}

impl TryFrom<u32> for ActivationType {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::SiLU),
            1 => Ok(Self::GELU),
            _ => Err(()),
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

pub fn activate<T: Float>(
    x: T,
    activation_type: &ActivationType,
) -> T {
    match activation_type {
        ActivationType::SiLU => silu(x),
        ActivationType::GELU => gelu(x),
    }
}
