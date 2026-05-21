use crate::{
    DataType,
    backends::common::{
        Allocation, Backend,
        gpu_types::{QuantizationMethod, QuantizationMode},
    },
};

#[derive(Debug, thiserror::Error)]
pub enum QuantizedMatmulError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unsupported group size: {0}")]
    UnsupportedGroupSize(usize),
    #[error("Hadamard not supported for this kernel configuration")]
    UnsupportedHadamard,
}

#[derive(Debug, Clone, Copy)]
pub struct QuantizedMatmulConfiguration {
    pub data_type: DataType,
    pub group_size: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub mode: QuantizationMode,
    pub quantization_method: QuantizationMethod,
    pub use_hadamard: bool,
}

pub struct QuantizedMatmulArguments<'a, B: Backend> {
    pub a: &'a Allocation<B>,
    pub a_offset: usize,
    pub b: &'a Allocation<B>,
    pub scales: &'a Allocation<B>,
    pub zero_points_or_biases: &'a Allocation<B>,
    pub output: &'a mut Allocation<B>,
    pub hadamard_factors: Option<&'a Allocation<B>>,
    pub batch_dim: usize,
}
