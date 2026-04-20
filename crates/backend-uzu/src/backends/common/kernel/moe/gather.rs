use crate::{
    DataType,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        kernel::{MoeGatherXPerm1DKernel, MoeGatherXPerm2DKernel},
    },
};

pub struct MoeGatherArguments<'a, B: Backend> {
    pub x: &'a Allocation<B>,
    pub bucketed_ids: &'a Allocation<B>,
    pub x_perm: &'a mut Allocation<B>,
    pub sumk: &'a Allocation<B>,
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
        match dtype {
            DataType::F32 => self.f32.encode(
                args.x,
                args.bucketed_ids,
                args.x_perm,
                args.sumk,
                args.d_model as u32,
                args.t as u32,
                args.k as u32,
                encoder,
            ),
            DataType::F16 => self.f16.encode(
                args.x,
                args.bucketed_ids,
                args.x_perm,
                args.sumk,
                args.d_model as u32,
                args.t as u32,
                args.k as u32,
                encoder,
            ),
            DataType::BF16 => self.bf16.encode(
                args.x,
                args.bucketed_ids,
                args.x_perm,
                args.sumk,
                args.d_model as u32,
                args.t as u32,
                args.k as u32,
                encoder,
            ),
            _ => panic!("Unsupported data type: {:?}", dtype),
        };
    }
}
