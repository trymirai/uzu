use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        kernel::{MoeGatherXPerm1DKernel, MoeGatherXPerm2DKernel},
    },
    data_type::DataType,
};

enum MoeGatherVariant<B: Backend> {
    OneD(<B::Kernels as Kernels>::MoeGatherXPerm1DKernel),
    TwoD(<B::Kernels as Kernels>::MoeGatherXPerm2DKernel),
}

pub struct MoeGather<B: Backend> {
    variant: MoeGatherVariant<B>,
    data_type: DataType,
}

impl<B: Backend> MoeGather<B> {
    pub fn new(
        ctx: &B::Context,
        data_type: DataType,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            variant: if data_type == DataType::BF16 {
                MoeGatherVariant::TwoD(<B::Kernels as Kernels>::MoeGatherXPerm2DKernel::new(ctx, data_type)?)
            } else {
                MoeGatherVariant::OneD(<B::Kernels as Kernels>::MoeGatherXPerm1DKernel::new(ctx, data_type)?)
            },
            data_type,
        })
    }

    pub fn encode(
        &self,
        input: &Allocation<B>,
        bucketed_ids: &Allocation<B>,
        sumk: &Allocation<B>,
        batch_dim: usize,
        num_active_experts: usize,
        d_model: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut x_perm =
            encoder.allocate_scratch(size_for_shape(&[batch_dim * num_active_experts, d_model], self.data_type))?;
        encoder.encode_fill(&mut x_perm, 0);

        match &self.variant {
            MoeGatherVariant::OneD(kernel) => kernel.encode(
                input,
                bucketed_ids,
                &mut x_perm,
                sumk,
                d_model as u32,
                batch_dim as u32,
                num_active_experts as u32,
                encoder,
            ),
            MoeGatherVariant::TwoD(kernel) => kernel.encode(
                input,
                bucketed_ids,
                &mut x_perm,
                sumk,
                d_model as u32,
                batch_dim as u32,
                num_active_experts as u32,
                encoder,
            ),
        };

        Ok(x_perm)
    }
}

#[cfg(test)]
#[path = "../../../../unit/encodable_block/moe/moe_gather_test.rs"]
mod tests;
