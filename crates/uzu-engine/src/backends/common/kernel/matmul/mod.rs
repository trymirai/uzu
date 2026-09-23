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
pub use matmul_b::{MatmulB, QuantizedB, QuantizedCorrection};
pub use routing::{ActivationFormat, MatmulShape};

use crate::backends::common::gpu_types::{QUANT_PARAMS_GROUP_OUTPUT_ALIGNMENT, QuantizationMode};

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

    pub const fn scale_shape(self) -> [u32; 2] {
        self.shape(1)
    }

    pub fn zero_point_shape(
        self,
        mode: QuantizationMode,
    ) -> [u32; 2] {
        self.shape(mode.packing_divisor())
    }

    pub const fn scale_strides(self) -> QuantParamsStrides {
        self.strides(1)
    }

    pub fn zero_point_strides(
        self,
        mode: QuantizationMode,
    ) -> QuantParamsStrides {
        self.strides(mode.packing_divisor())
    }

    const fn shape(
        self,
        packing_divisor: u32,
    ) -> [u32; 2] {
        let row_values = self.padded_values_per_row(packing_divisor);
        match self.layout {
            QuantParamsLayout::OutputGroup => [self.output_count, row_values / packing_divisor],
            QuantParamsLayout::GroupOutput => [self.group_count, row_values / packing_divisor],
        }
    }

    const fn strides(
        self,
        packing_divisor: u32,
    ) -> QuantParamsStrides {
        let row_values = self.padded_values_per_row(packing_divisor);
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

    const fn padded_values_per_row(
        self,
        packing_divisor: u32,
    ) -> u32 {
        match self.layout {
            QuantParamsLayout::OutputGroup => self.group_count.next_multiple_of(packing_divisor),
            QuantParamsLayout::GroupOutput => self.output_count.next_multiple_of(QUANT_PARAMS_GROUP_OUTPUT_ALIGNMENT),
        }
    }
}
