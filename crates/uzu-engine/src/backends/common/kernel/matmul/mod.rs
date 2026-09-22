mod arguments;
mod d_ops;
mod error;
mod kernel;
mod matmul_a;
mod matmul_b;
pub mod routing;

pub use arguments::MatmulArguments;
pub use d_ops::MatmulDOps;
pub use error::MatmulError;
pub use kernel::MatmulKernel;
pub use matmul_a::{Int8CodeLayout, MatmulA};
pub use matmul_b::MatmulB;
pub use routing::{ActivationFormat, MatmulShape};

const GROUP_OUTPUT_ALIGNMENT: u32 = 4;

#[derive(Clone, Copy, PartialEq)]
pub enum QuantParamsLayout {
    OutputGroup,
    GroupOutput,
}

impl QuantParamsLayout {
    pub const fn row_stride(
        self,
        columns: u32,
        groups: u32,
    ) -> u32 {
        match self {
            Self::OutputGroup => groups,
            Self::GroupOutput => self.group_stride(columns),
        }
    }

    pub const fn group_stride(
        self,
        columns: u32,
    ) -> u32 {
        match self {
            Self::OutputGroup => 1,
            Self::GroupOutput => columns.next_multiple_of(GROUP_OUTPUT_ALIGNMENT),
        }
    }

    pub const fn plane_shape(
        self,
        columns: u32,
        groups: u32,
        packing_divisor: u32,
    ) -> [u32; 2] {
        let (rows, row_width) = match self {
            Self::OutputGroup => (columns, groups),
            Self::GroupOutput => (groups, self.group_stride(columns)),
        };
        [rows, row_width.div_ceil(packing_divisor)]
    }
}
