use crate::backends::common::{Allocation, Backend};

pub enum MatmulA<'a, B: Backend> {
    FullPrecision {
        values: &'a Allocation<B>,
        offset: usize,
    },
    /// Groupwise symmetric int8 activations. The quantized weight buffer must
    /// hold sign-converted codes (unsigned code XOR midpoint); see
    /// `LinearMatmul::sign_convert_quantized_weights_for_int8_activations`.
    Int8Symmetric {
        values: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        group_size: u32,
    },
}

impl<'a, B: Backend> MatmulA<'a, B> {
    pub fn is_int8(&self) -> bool {
        matches!(self, Self::Int8Symmetric { .. })
    }
}
