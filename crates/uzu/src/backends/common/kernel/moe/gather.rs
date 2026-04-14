use crate::{
    DataType,
    backends::common::{
        Backend, Encoder, Kernels,
        kernel::{BufferArg, BufferArgMut, MoeGatherXPerm1DKernel, MoeGatherXPerm2DKernel},
    },
};

#[derive(Debug)]
pub struct MoeGatherArguments<'a, B: Backend> {
    pub x_buffer: &'a B::Buffer,
    pub bucketed_ids_buffer: &'a B::Buffer,
    pub x_perm_buffer: &'a mut B::Buffer,
    pub sumk_buffer: &'a B::Buffer,
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
        encoder: &mut Encoder<B>,
        dtype: DataType,
        args: MoeGatherArguments<B>,
    ) {
        self.encode_with_offsets(
            encoder,
            dtype,
            args.x_buffer,
            args.bucketed_ids_buffer,
            args.x_perm_buffer,
            args.sumk_buffer,
            args.t,
            args.k,
            args.d_model,
        )
    }

    pub fn encode_with_offsets<'x, 'bucketed_ids, 'x_perm, 'sumk>(
        &self,
        encoder: &mut Encoder<B>,
        dtype: DataType,
        x_buffer: impl BufferArg<'x, B::Buffer>,
        bucketed_ids_buffer: impl BufferArg<'bucketed_ids, B::Buffer>,
        x_perm_buffer: impl BufferArgMut<'x_perm, B::Buffer>,
        sumk_buffer: impl BufferArg<'sumk, B::Buffer>,
        t: usize,
        k: usize,
        d_model: usize,
    ) {
        match dtype {
            DataType::F32 => self.f32.encode(
                x_buffer,
                bucketed_ids_buffer,
                x_perm_buffer,
                sumk_buffer,
                d_model as u32,
                t as u32,
                k as u32,
                encoder,
            ),
            DataType::F16 => self.f16.encode(
                x_buffer,
                bucketed_ids_buffer,
                x_perm_buffer,
                sumk_buffer,
                d_model as u32,
                t as u32,
                k as u32,
                encoder,
            ),
            DataType::BF16 => self.bf16.encode(
                x_buffer,
                bucketed_ids_buffer,
                x_perm_buffer,
                sumk_buffer,
                d_model as u32,
                t as u32,
                k as u32,
                encoder,
            ),
            _ => panic!("Unsupported data type: {:?}", dtype),
        };
    }
}
