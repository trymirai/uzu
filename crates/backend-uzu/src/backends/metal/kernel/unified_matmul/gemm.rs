//! Isolated unified GEMM implementation.

use crate::{
    DataType,
    backends::{
        common::{
            Backend, Encoder,
            kernel::{BufferArg, BufferArgMut},
        },
        metal::{
            Metal,
            context::MetalContext,
            error::MetalError,
            kernel::UnifiedGemmSimdgroupFullPrecisionMetalKernel,
        },
    },
};

#[allow(dead_code)]
pub(crate) struct UnifiedGemmKernel {
    full_precision: UnifiedGemmSimdgroupFullPrecisionMetalKernel,
}

#[allow(dead_code)]
impl UnifiedGemmKernel {
    pub(crate) fn new(
        context: &MetalContext,
        data_type: DataType,
    ) -> Result<Self, MetalError> {
        let full_precision = UnifiedGemmSimdgroupFullPrecisionMetalKernel::new(context, data_type)?;
        Ok(Self {
            full_precision,
        })
    }

    pub(crate) fn encode<'a, 'b, 'd>(
        &self,
        a: impl BufferArg<'a, <Metal as Backend>::Buffer>,
        b: impl BufferArg<'b, <Metal as Backend>::Buffer>,
        d: impl BufferArgMut<'d, <Metal as Backend>::Buffer>,
        group_count_x: u32,
        group_count_y: u32,
        encoder: &mut Encoder<Metal>,
    ) {
        self.full_precision.encode(a, b, d, group_count_x, group_count_y, encoder);
    }
}

