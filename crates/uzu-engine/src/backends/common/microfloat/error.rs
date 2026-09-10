use thiserror::Error;

use super::MicrofloatFormat;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MicrofloatError {
    #[error("unsupported {format:?} bit width: {bits}")]
    UnsupportedBits {
        format: MicrofloatFormat,
        bits: u32,
    },
    #[error("unsupported {format:?} group size: {group_size}")]
    UnsupportedGroupSize {
        format: MicrofloatFormat,
        group_size: u32,
    },
    #[error("microfloat rows and columns must be nonzero")]
    EmptyShape,
    #[error("microfloat columns {columns} are not divisible by group size {group_size}")]
    MisalignedColumns {
        columns: u32,
        group_size: u32,
    },
    #[error("microfloat storage size overflows usize")]
    SizeOverflow,
}
