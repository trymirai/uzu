/// Error type for DLTensor conversion operations
#[derive(Debug, thiserror::Error)]
pub enum DLTensorError {
    #[error("Unsupported device type: expected Metal (8), got {0}")]
    UnsupportedDevice(i32),

    #[error("Unsupported data type: code={code}, bits={bits}, lanes={lanes}")]
    UnsupportedDataType {
        code: u8,
        bits: u8,
        lanes: u16,
    },

    #[error("Non-contiguous tensors are not supported (strides must be NULL)")]
    NonContiguous,

    #[error("Invalid shape: ndim={ndim} but shape pointer is NULL")]
    InvalidShape {
        ndim: i32,
    },

    #[error("Vectorized data types (lanes > 1) are not supported")]
    VectorizedType,

    #[error("Data pointer is NULL")]
    NullDataPointer,
}
