pub mod common;
mod dispatch_descriptor;
mod gemm;
mod gemv;
mod kernel;
mod split_k;

pub use common::MatmulArguments;
pub use dispatch_descriptor::{MatmulKernelVariant, determine_kernel_variant};
pub use gemv::GemvKernel;
pub use kernel::MatmulKernel;
use metal::MTLComputeCommandEncoder;
use objc2::runtime::ProtocolObject;
pub use split_k::SplitKGemm;

use crate::{
    DataType,
    backends::{
        common::kernel::matmul::{
            FullPrecisionMatmulArguments, FullPrecisionMatmulKernel as FullPrecisionMatmulKernelTrait,
        },
        metal::{Metal, context::MetalContext, error::MetalError},
    },
};

impl FullPrecisionMatmulKernelTrait for MatmulKernel {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        data_type: DataType,
    ) -> Result<Self, MetalError> {
        let mut kernel = MatmulKernel::new(data_type)?;
        kernel.precompile(context)?;
        Ok(kernel)
    }

    fn encode(
        &mut self,
        context: &MetalContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        arguments: FullPrecisionMatmulArguments<Metal>,
    ) {
        let matmul_arguments = MatmulArguments {
            a: arguments.a,
            a_offset: arguments.a_offset as u64,
            b: arguments.b,
            c: None,
            d: arguments.output,
            bias: arguments.bias,
            batch: arguments.batch as i32,
            input_dim: arguments.input_dim as i32,
            output_dim: arguments.output_dim as i32,
            lda: arguments.input_dim as i32,
            ldb: arguments.input_dim as i32,
            ldd: arguments.output_dim as i32,
            batch_count: 1,
            alpha: 1.0,
            beta: 0.0,
            transpose_a: false,
            transpose_b: true,
        };

        MatmulKernel::encode(self, context, encoder, matmul_arguments)
            .expect("Failed to encode full precision matmul kernel");
    }
}
