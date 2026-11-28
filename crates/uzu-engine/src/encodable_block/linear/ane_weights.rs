use crate::{
    backends::common::{Backend, Buffer, kernel::matmul::MatmulB},
    data_type::DataType,
    encodable_block::weight_matrix::WeightMatrix,
};

/// A compact quantized projection that can be staged into reusable ANE rows.
pub struct AneWeightLayer<'a, B: Backend> {
    pub matrix: &'a WeightMatrix<B>,
    pub rows: u32,
    pub columns: u32,
    pub input_signs: &'a B::GlobalGpuBuffer,
    pub input_hadamard_size: u32,
    pub output_signs: Option<&'a B::GlobalGpuBuffer>,
    pub output_hadamard_size: Option<u32>,
}

impl<'a, B: Backend> AneWeightLayer<'a, B> {
    pub fn new(
        matrix: &'a WeightMatrix<B>,
        rows: u32,
        columns: u32,
        weights_data_type: DataType,
        has_biases: bool,
        input_signs: &'a B::GlobalGpuBuffer,
        input_hadamard_size: u32,
        output_signs: Option<&'a B::GlobalGpuBuffer>,
        output_hadamard_size: Option<u32>,
    ) -> Option<Self> {
        if !matches!(input_hadamard_size, 32 | 128)
            || input_signs.size() == 0
            || columns == 0
            || rows == 0
            || weights_data_type != DataType::BF16
            || output_hadamard_size.is_some_and(|size| size != 32)
            || output_signs.is_some() != output_hadamard_size.is_some()
        {
            return None;
        }
        if !matches!(matrix.matmul_b(), MatmulB::Quantized(_)) || has_biases {
            return None;
        }
        Some(Self {
            matrix,
            rows,
            columns,
            input_signs,
            input_hadamard_size,
            output_signs,
            output_hadamard_size,
        })
    }
}
