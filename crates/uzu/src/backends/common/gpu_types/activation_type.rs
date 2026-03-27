use num_traits::Float;

const GELU_K0: f32 = 0.044715;
const GELU_K1: f32 = 0.7978845608; // sqrt(2/pi)

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ActivationType {
    SILU {
        alpha: f32,
    },
    GELU,
    TANH,
    IDENTITY,
}

impl ActivationType {
    pub fn silu_default() -> ActivationType {
        ActivationType::silu(1.0f32)
    }

    pub fn silu(alpha: f32) -> ActivationType {
        ActivationType::SILU {
            alpha,
        }
    }

    pub fn activate<T: Float>(
        &self,
        x: T,
    ) -> T {
        match self {
            ActivationType::SILU {
                alpha,
            } => silu(x, *alpha),
            ActivationType::GELU => gelu(x),
            ActivationType::TANH => tanh_activation(x),
            ActivationType::IDENTITY => x,
        }
    }
}

fn silu<T: Float>(
    x: T,
    alpha: f32,
) -> T {
    let x_float = x.to_f32().unwrap();
    let y_float = x_float / (1.0f32 + (-alpha * x_float).exp());
    T::from(y_float).unwrap()
}

fn gelu<T: Float>(x: T) -> T {
    let x_float = x.to_f32().unwrap();
    let tan_arg = GELU_K1 * (x_float + GELU_K0 * x_float * x_float * x_float);
    let y_float = 0.5f32 * x_float * (1.0f32 + tan_arg.tanh());
    T::from(y_float).unwrap()
}

fn tanh_activation<T: Float>(x: T) -> T {
    let x_float = x.to_f32().unwrap();
    T::from(x_float.tanh()).unwrap()
}
