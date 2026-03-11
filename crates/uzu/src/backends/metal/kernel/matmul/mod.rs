mod dispatch_descriptor;
mod gemm_mpp;
pub use dispatch_descriptor::choose_dispatch_descriptor;

use crate::{
    DataType,
    backends::{
        common::{
            Backend, CommandBuffer,
            kernel::matmul::{
                FullPrecisionMatmulArguments, FullPrecisionMatmulKernel as FullPrecisionMatmulKernelTrait,
                MatmulArguments, MatmulError, MatmulKernel,
            },
        },
        metal::{Metal, context::MetalContext},
    },
};

impl FullPrecisionMatmulKernelTrait for MatmulKernel<Metal> {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        data_type: DataType,
    ) -> Result<Self, MatmulError<Metal>> {
        let mut kernel = MatmulKernel::new(data_type)?;
        kernel.precompile(context)?;
        Ok(kernel)
    }

    fn encode(
        &mut self,
        context: &MetalContext,
        encoder: &mut <<Metal as Backend>::CommandBuffer as CommandBuffer>::Encoding,
        arguments: FullPrecisionMatmulArguments<Metal>,
    ) {
        let mut matmul_arguments = MatmulArguments {
            a: arguments.a,
            a_offset: arguments.a_offset as u64,
            b: arguments.b,
            d: arguments.output,
            bias: arguments.bias,
            batch: arguments.batch as i32,
            input_dim: arguments.input_dim as i32,
            output_dim: arguments.output_dim as i32,
            leading_dim_a: arguments.input_dim as i32,
            leading_dim_b: arguments.input_dim as i32,
            leading_dim_d: arguments.output_dim as i32,
            batch_count: 1,
            transpose_b: true,
        };

        MatmulKernel::<Metal>::apply_batch_collapse(&mut matmul_arguments);

        let descriptor = dispatch_descriptor::choose_dispatch_descriptor(
            context,
            self.data_type,
            &matmul_arguments,
        )
        .expect("Failed to create dispatch descriptor for full precision matmul");

        self.encode_with_descriptor(context, matmul_arguments, &descriptor, encoder)
            .expect("Failed to encode full precision matmul kernel");
    }
}
