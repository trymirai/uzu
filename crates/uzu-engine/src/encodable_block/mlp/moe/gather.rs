use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
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
        input: impl BufferRef<Backend = B>,
        bucketed_ids: impl BufferRef<Backend = B>,
        sumk: impl BufferRef<Backend = B>,
        batch_dim: u32,
        num_active_experts: u32,
        d_model: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut x_perm =
            command_buffer.allocate_scratch_for_shape(&[batch_dim, num_active_experts, d_model], self.data_type)?;
        command_buffer.encode_fill(&mut x_perm, 0);

        match &self.variant {
            MoeGatherVariant::OneD(kernel) => kernel.encode(
                input,
                bucketed_ids,
                &mut x_perm,
                sumk,
                d_model,
                batch_dim,
                num_active_experts,
                command_buffer,
            ),
            MoeGatherVariant::TwoD(kernel) => kernel.encode(
                input,
                bucketed_ids,
                &mut x_perm,
                sumk,
                d_model,
                batch_dim,
                num_active_experts,
                command_buffer,
            ),
        };

        Ok(x_perm)
    }
}

#[cfg(test)]
#[path = "../../../../unit/encodable_block/moe/moe_gather_test.rs"]
mod tests;
