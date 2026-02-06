use objc2::Message;

use crate::backends::{
    common::kernel::MoeRouterTopKKernel,
    metal::{
        KernelDataType, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
        MTLContext, MTLError, ProtocolObject, Retained,
        kernel::dsl::MoeRouterTopKMetalKernel,
    },
};

const MAX_EXPERTS: usize = 512;
const MAX_TOPK: usize = 128;

#[derive(Debug, thiserror::Error)]
pub enum MoeRouterTopKError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("Invalid configuration: T={t}, d_model={d_model}, E={e}, K={k}")]
    InvalidConfig {
        t: usize,
        d_model: usize,
        e: usize,
        k: usize,
    },
}

#[derive(Debug)]
pub struct MoeRouterTopKArguments<'a> {
    pub input_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [T, d_model]
    pub weight_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [E, d_model]
    pub bias_buffer: &'a ProtocolObject<dyn MTLBuffer>,  // [E]
    pub topk_ids_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [T, K]
    pub topk_probs_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [T, K]
    pub t: usize,
    pub d_model: usize,
    pub e: usize,
    pub k: usize,
    pub renorm: bool,
}

pub struct MoeRouterTopKKernelWrapper {
    kernel: MoeRouterTopKMetalKernel,
}

impl MoeRouterTopKKernelWrapper {
    pub fn new(
        ctx: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, MoeRouterTopKError> {
        Ok(Self {
            kernel: MoeRouterTopKMetalKernel::new(
                ctx,
                data_type.into(),
                data_type.into(),
            )?,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        args: &MoeRouterTopKArguments,
    ) -> Result<(), MoeRouterTopKError> {
        if args.t == 0 || args.e == 0 || args.k == 0 {
            return Ok(());
        }
        if args.d_model % 4 != 0 {
            return Err(MoeRouterTopKError::InvalidConfig {
                t: args.t,
                d_model: args.d_model,
                e: args.e,
                k: args.k,
            });
        }
        if args.e > MAX_EXPERTS || args.k > MAX_TOPK {
            return Err(MoeRouterTopKError::InvalidConfig {
                t: args.t,
                d_model: args.d_model,
                e: args.e,
                k: args.k,
            });
        }

        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        self.kernel.encode(
            &args.input_buffer.retain(),
            &args.weight_buffer.retain(),
            &args.bias_buffer.retain(),
            &args.topk_ids_buffer.retain(),
            &args.topk_probs_buffer.retain(),
            args.t as u32,
            args.d_model as u32,
            args.e as u32,
            args.k as u32,
            args.renorm,
            &encoder,
        );
        encoder.end_encoding();

        Ok(())
    }
}
