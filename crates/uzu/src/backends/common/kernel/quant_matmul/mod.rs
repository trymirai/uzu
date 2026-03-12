mod arguments;
mod configuration;
mod kernel;
mod quantization_type;
mod variant;

pub use arguments::QuantizedMatmulArguments;
pub use configuration::QuantizedMatmulConfiguration;
pub use kernel::QuantizedMatmulKernelEncodable;
pub use quantization_type::QuantizedMatmulType;

use crate::{DataType, backends::common::Backend};

#[derive(Debug, thiserror::Error)]
pub enum QuantizedMatmulError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unsupported group size: {0}")]
    UnsupportedGroupSize(usize),
    #[error("Unsupported bits: {0}")]
    UnsupportedBits(usize),
    #[error("Value `{name}` does not fit i32: {value}")]
    ValueOutOfRange {
        name: &'static str,
        value: usize,
    },
    #[error("Quantization type mismatch: kernel={kernel:?}, args={args:?}")]
    QuantizationTypeMismatch {
        kernel: QuantizedMatmulType,
        args: QuantizedMatmulType,
    },
    #[error("Missing kernel for key: {0}")]
    MissingKernel(Box<str>),
}
