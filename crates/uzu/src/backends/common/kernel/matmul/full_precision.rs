use super::{MatmulArguments, MatmulError, MatmulKernel, MatmulKernels};
use crate::{
    DataType,
    backends::common::{
        Backend, CommandBuffer, kernel::matmul::dispatch_descriptor::choose_matmul_dispatch_descriptor,
    },
};

pub struct FullPrecisionMatmulArguments<'a, B: Backend> {
    pub a: &'a B::Buffer,
    pub a_offset: usize,
    pub b: &'a B::Buffer,
    pub output: &'a mut B::Buffer,
    pub bias: Option<&'a B::Buffer>,
    pub batch: usize,
    pub input_dim: usize,
    pub output_dim: usize,
}

pub trait FullPrecisionMatmulKernel: Sized {
    type Backend: Backend<Kernels: MatmulKernels<FullPrecisionMatmulKernel = Self>>;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        data_type: DataType,
    ) -> Result<Self, MatmulError<Self::Backend>>;

    fn encode(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        command_buffer: &mut <<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding,
        arguments: FullPrecisionMatmulArguments<Self::Backend>,
    );
}

impl<B: Backend<Kernels: MatmulKernels<FullPrecisionMatmulKernel = Self>>> FullPrecisionMatmulKernel
    for MatmulKernel<B>
{
    type Backend = B;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        data_type: DataType,
    ) -> Result<Self, MatmulError<Self::Backend>> {
        let mut kernel = MatmulKernel::new(data_type)?;
        kernel.precompile(context)?;
        Ok(kernel)
    }

    fn encode(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        command_buffer: &mut <<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding,
        arguments: FullPrecisionMatmulArguments<Self::Backend>,
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
            lda: arguments.input_dim as i32,
            ldb: arguments.input_dim as i32,
            ldd: arguments.output_dim as i32,
            batch_count: 1,
            transpose_b: true,
        };

        MatmulKernel::<B>::apply_batch_collapse(&mut matmul_arguments);

        let descriptor = choose_matmul_dispatch_descriptor(context, self.data_type, &matmul_arguments)
            .expect("Failed to create dispatch descriptor for full precision matmul");

        self.encode_with_descriptor(context, matmul_arguments, &descriptor, command_buffer)
            .expect("Failed to encode full precision matmul kernel");
    }
}
