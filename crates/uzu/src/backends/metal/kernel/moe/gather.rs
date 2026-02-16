use metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder};
use objc2::{Message, rc::Retained, runtime::ProtocolObject};

use crate::{
    DataType,
    backends::{
        common::kernel::{MoeGatherXPerm1DKernel, MoeGatherXPerm2DKernel},
        metal::{
            MetalContext, MetalError,
            kernel::dsl::{MoeGatherXPerm1DMetalKernel, MoeGatherXPerm2DMetalKernel},
        },
    },
};

#[derive(Debug)]
pub struct MoeGatherArguments<'a> {
    pub x_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub bucketed_ids_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub x_perm_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub sumk_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub t: usize,
    pub k: usize,
    pub d_model: usize,
}

pub struct MoeGatherKernels {
    bf16: MoeGatherXPerm2DMetalKernel,
    f16: MoeGatherXPerm1DMetalKernel,
    f32: MoeGatherXPerm1DMetalKernel,
}

impl MoeGatherKernels {
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        Ok(Self {
            bf16: MoeGatherXPerm2DMetalKernel::new(ctx, DataType::BF16)?,
            f16: MoeGatherXPerm1DMetalKernel::new(ctx, DataType::F16)?,
            f32: MoeGatherXPerm1DMetalKernel::new(ctx, DataType::F32)?,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        dtype: DataType,
        args: &MoeGatherArguments,
    ) {
        let encoder = command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
        match dtype {
            DataType::F32 => self.f32.encode(
                &args.x_buffer.retain(),
                &args.bucketed_ids_buffer.retain(),
                &args.x_perm_buffer.retain(),
                &args.sumk_buffer.retain(),
                args.d_model as u32,
                args.t as u32,
                args.k as u32,
                &encoder,
            ),
            DataType::F16 => self.f16.encode(
                &args.x_buffer.retain(),
                &args.bucketed_ids_buffer.retain(),
                &args.x_perm_buffer.retain(),
                &args.sumk_buffer.retain(),
                args.d_model as u32,
                args.t as u32,
                args.k as u32,
                &encoder,
            ),
            DataType::BF16 => self.bf16.encode(
                &args.x_buffer.retain(),
                &args.bucketed_ids_buffer.retain(),
                &args.x_perm_buffer.retain(),
                &args.sumk_buffer.retain(),
                args.d_model as u32,
                args.t as u32,
                args.k as u32,
                &encoder,
            ),
            _ => panic!("Unsupported data type: {:?}", dtype),
        }
        encoder.end_encoding();
    }
}
