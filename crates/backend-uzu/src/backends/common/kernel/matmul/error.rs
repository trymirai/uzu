use thiserror::Error;

use crate::{
    backends::common::{Backend, gpu_types::gemm::GemmDTransform},
    data_type::DataType,
};

#[derive(Debug, Error)]
pub enum MatmulError<B: Backend> {
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unsupported group size: {0}")]
    UnsupportedGroupSize(usize),
    #[error("Unsupported D-transform op {bit:?} on path {path}")]
    UnsupportedDOp {
        bit: GemmDTransform,
        path: &'static str,
    },
    #[error("Unsupported B layout on path {path}")]
    UnsupportedLayout {
        path: &'static str,
    },
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
}
