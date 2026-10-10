mod arguments;
mod d_ops;
mod error;
mod kernel;
mod matmul_a;
mod matmul_b;
mod output;
pub mod routing;

pub use arguments::MatmulArguments;
pub use d_ops::MatmulDOps;
pub use error::MatmulError;
pub use kernel::MatmulKernel;
pub use matmul_a::{Int8CodeLayout, MatmulA};
pub use matmul_b::{MatmulB, QuantizedB, QuantizedCorrection, TrellisFormat};
pub use output::MatmulOutput;
pub use routing::{ActivationFormat, MatmulShape};

use crate::backends::common::gpu_types::{QUANT_PARAMS_GROUP_OUTPUT_ALIGNMENT, QuantizationMode};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuantParamsLayout {
    /// `[N, G]`.
    OutputGroup,
    /// `[G, round_up_4(N)]`.
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
    row_count: u32,
    group_count: u32,
}

impl QuantParams {
    pub const fn new(
        layout: QuantParamsLayout,
        row_count: u32,
        group_count: u32,
    ) -> Self {
        Self {
            layout,
            row_count,
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
            QuantParamsLayout::OutputGroup => [self.row_count, row_values / packing_divisor],
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
            QuantParamsLayout::GroupOutput => self.row_count.next_multiple_of(QUANT_PARAMS_GROUP_OUTPUT_ALIGNMENT),
        }
    }
}

/// Packs complete N=32, K=128 W4 tiles for four contiguous words per SIMD lane.
/// Values and nibble order within each word are unchanged.
#[cfg(backend = "metal")]
pub(crate) fn pack_mxu_w4_tiles(
    source: &[u32],
    destination: &mut [u32],
    n: u32,
    k: u32,
) {
    assert!(n.is_multiple_of(32) && k.is_multiple_of(128));
    assert_eq!(source.len(), n as usize * k as usize / 8);
    assert_eq!(destination.len(), source.len());
    let groups = k as usize / 128;
    let row_words = k as usize / 8;
    for block in 0..n as usize / 32 {
        for batch in 0..groups {
            for word in 0..4 {
                for lane in 0..32 {
                    let quad = lane / 4;
                    let row = (quad & 4) + (lane / 2) % 4;
                    let column = (quad & 2) + lane % 2;
                    for chunk in 0..4 {
                        destination[(((block * groups + batch) * 4 + word) * 32 + lane) * 4 + chunk] =
                            source[(block * 32 + row + word * 8) * row_words + batch * 16 + chunk * 4 + column];
                    }
                }
            }
        }
    }
}
