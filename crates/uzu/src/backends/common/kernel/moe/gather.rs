use crate::{
    DataType,
    backends::common::{
        Backend, CommandBuffer, Kernels,
        kernel::{MoeGatherXPerm1DKernel, MoeGatherXPerm2DKernel},
    },
};

#[derive(Debug)]
pub struct MoeGatherArguments<'a, B: Backend> {
    pub x_buffer: &'a B::NativeBuffer,
    pub bucketed_ids_buffer: &'a B::NativeBuffer,
    pub x_perm_buffer: &'a B::NativeBuffer,
    pub sumk_buffer: &'a B::NativeBuffer,
    pub t: usize,
    pub k: usize,
    pub d_model: usize,
}

pub struct MoeGatherKernels<B: Backend> {
    bf16: <B::Kernels as Kernels>::MoeGatherXPerm2DKernel,
    f16: <B::Kernels as Kernels>::MoeGatherXPerm1DKernel,
    f32: <B::Kernels as Kernels>::MoeGatherXPerm1DKernel,
}

impl<B: Backend> MoeGatherKernels<B> {
    pub fn new(ctx: &B::Context) -> Result<Self, B::Error> {
        Ok(Self {
            bf16: <B::Kernels as Kernels>::MoeGatherXPerm2DKernel::new(ctx, DataType::BF16)?,
            f16: <B::Kernels as Kernels>::MoeGatherXPerm1DKernel::new(ctx, DataType::F16)?,
            f32: <B::Kernels as Kernels>::MoeGatherXPerm1DKernel::new(ctx, DataType::F32)?,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &B::CommandBuffer,
        dtype: DataType,
        args: &MoeGatherArguments<B>,
    ) {
        command_buffer.with_compute_encoder(|encoder| match dtype {
            DataType::F32 => self.f32.encode(
                args.x_buffer,
                args.bucketed_ids_buffer,
                args.x_perm_buffer,
                args.sumk_buffer,
                args.d_model as u32,
                args.t as u32,
                args.k as u32,
                encoder,
            ),
            DataType::F16 => self.f16.encode(
                args.x_buffer,
                args.bucketed_ids_buffer,
                args.x_perm_buffer,
                args.sumk_buffer,
                args.d_model as u32,
                args.t as u32,
                args.k as u32,
                encoder,
            ),
            DataType::BF16 => self.bf16.encode(
                args.x_buffer,
                args.bucketed_ids_buffer,
                args.x_perm_buffer,
                args.sumk_buffer,
                args.d_model as u32,
                args.t as u32,
                args.k as u32,
                encoder,
            ),
            _ => panic!("Unsupported data type: {:?}", dtype),
        });
    }
}
