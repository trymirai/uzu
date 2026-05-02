use crate::backends::metal::kernel::unified_matmul::gemm::{BitsPerWeight, GroupSize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct QuantizationParams {
    pub bits: BitsPerWeight,
    pub group_size: GroupSize,
}
