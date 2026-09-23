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

use crate::{backends::common::gpu_types::QUANT_PARAMS_GROUP_OUTPUT_ALIGNMENT, data_type::DataType};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuantParamsLayout {
    OutputGroup,
    GroupOutput,
}

#[derive(Clone, Copy, Default)]
pub struct QuantParamsStrides {
    pub output_stride: u32,
    pub group_stride: u32,
}

#[derive(Clone, Copy)]
pub struct QuantParams {
    layout: QuantParamsLayout,
    output_count: u32,
    group_count: u32,
}

impl QuantParams {
    pub const fn new(
        layout: QuantParamsLayout,
        output_count: u32,
        group_count: u32,
    ) -> Self {
        Self {
            layout,
            output_count,
            group_count,
        }
    }

    pub const fn layout(self) -> QuantParamsLayout {
        self.layout
    }

    pub const fn shape(
        self,
        data_type: DataType,
    ) -> [u32; 2] {
        let packing_divisor = Self::packing_divisor(data_type);
        let row_values = self.padded_values_per_row(data_type);
        match self.layout {
            QuantParamsLayout::OutputGroup => [self.output_count, row_values / packing_divisor],
            QuantParamsLayout::GroupOutput => [self.group_count, row_values / packing_divisor],
        }
    }

    pub const fn strides(
        self,
        data_type: DataType,
    ) -> QuantParamsStrides {
        let row_values = self.padded_values_per_row(data_type);
        match self.layout {
            QuantParamsLayout::OutputGroup => QuantParamsStrides {
                output_stride: row_values,
                group_stride: 1,
            },
            QuantParamsLayout::GroupOutput => QuantParamsStrides {
                output_stride: 1,
                group_stride: row_values,
            },
        }
    }

    pub const fn storage_type(
        self,
        data_type: DataType,
    ) -> DataType {
        match data_type {
            DataType::U4 => DataType::U8,
            _ => data_type,
        }
    }

    pub const fn index(
        self,
        data_type: DataType,
        output: u32,
        group: u32,
    ) -> u32 {
        let strides = self.strides(data_type);
        output * strides.output_stride + group * strides.group_stride
    }

    const fn packing_divisor(data_type: DataType) -> u32 {
        match data_type {
            DataType::U4 => 2,
            _ => 1,
        }
    }

    const fn padded_values_per_row(
        self,
        data_type: DataType,
    ) -> u32 {
        match self.layout {
            QuantParamsLayout::OutputGroup => self.group_count.next_multiple_of(Self::packing_divisor(data_type)),
            QuantParamsLayout::GroupOutput => self.output_count.next_multiple_of(QUANT_PARAMS_GROUP_OUTPUT_ALIGNMENT),
        }
    }
}
