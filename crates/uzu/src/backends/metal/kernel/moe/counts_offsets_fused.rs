use crate::backends::metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLContext, MTLError, MTLSize, ProtocolObject, Retained,
    metal_extensions::ComputeEncoderSetValue,
};

// ---- Fused Counts + Offsets Kernel ----

#[derive(Debug, thiserror::Error)]
pub enum MoeCountsOffsetsFusedError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("Invalid dimensions: T={t}, E={e}, K={k}")]
    InvalidDimensions {
        t: usize,
        e: usize,
        k: usize,
    },
}

pub struct MoeCountsOffsetsFusedKernel {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

#[derive(Debug)]
pub struct MoeCountsOffsetsFusedArguments<'a> {
    pub topk_ids_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub offsets_buffer: &'a ProtocolObject<dyn MTLBuffer>, // output [E+1]
    pub sum_k_buffer: &'a ProtocolObject<dyn MTLBuffer>,   // output [1]
    pub partials_buffer: &'a ProtocolObject<dyn MTLBuffer>, // output [num_tiles * 512] (for block_bases)
    pub t: usize,
    pub e: usize,
    pub k: usize,
}

impl MoeCountsOffsetsFusedKernel {
    pub fn new(
        mtl_context: &MTLContext
    ) -> Result<Self, MoeCountsOffsetsFusedError> {
        let pipeline = mtl_context
            .compute_pipeline_state("moe_counts_offsets_fused", None)?;
        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        args: MoeCountsOffsetsFusedArguments,
    ) -> Result<(), MoeCountsOffsetsFusedError> {
        if args.k == 0 || args.e == 0 {
            return Ok(());
        }
        if args.t == 0 {
            return Ok(());
        }
        if args.e < 1 {
            return Err(MoeCountsOffsetsFusedError::InvalidDimensions {
                t: args.t,
                e: args.e,
                k: args.k,
            });
        }

        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        encoder.set_compute_pipeline_state(&self.pipeline);

        encoder.set_buffer(Some(args.topk_ids_buffer), 0, 0);
        encoder.set_buffer(Some(args.offsets_buffer), 0, 1);
        encoder.set_buffer(Some(args.sum_k_buffer), 0, 2);
        encoder.set_buffer(Some(args.partials_buffer), 0, 3);

        let t_u32 = args.t as u32;
        let e_u32 = args.e as u32;
        let k_u32 = args.k as u32;

        encoder.set_value(&t_u32, 4);
        encoder.set_value(&e_u32, 5);
        encoder.set_value(&k_u32, 6);

        let threads_per_threadgroup = MTLSize::new(128, 1, 1);
        let tg = MTLSize::new(1, 1, 1);
        encoder.dispatch_threadgroups(tg, threads_per_threadgroup);

        encoder.end_encoding();
        Ok(())
    }
}
