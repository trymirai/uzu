pub mod common;
mod dispatch_descriptor;
mod gemm;

pub use dispatch_descriptor::choose_dispatch_descriptor;

use crate::{
    DataType,
    backends::{
        common::kernel::matmul::{
            FullPrecisionMatmulArguments, FullPrecisionMatmulKernel as FullPrecisionMatmulKernelTrait, MatmulArguments,
            MatmulKernel,
        },
        metal::{Metal, context::MetalContext, error::MetalError},
    },
};

impl FullPrecisionMatmulKernelTrait for MatmulKernel<Metal> {
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
        encoder: &<Metal as crate::backends::common::Backend>::ComputeEncoder,
        arguments: FullPrecisionMatmulArguments<Metal>,
    ) {
        let mut matmul_arguments = MatmulArguments {
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

        MatmulKernel::<Metal>::apply_batch_collapse(&mut matmul_arguments);

        let descriptor = choose_dispatch_descriptor(context, self.data_type, &matmul_arguments)
            .expect("Failed to create dispatch descriptor for full precision matmul");

        self.encode_with_descriptor(context, matmul_arguments, &descriptor, encoder)
            .expect("Failed to encode full precision matmul kernel");
    }
}
