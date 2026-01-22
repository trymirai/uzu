use std::{ffi::c_void, mem::size_of, ptr::NonNull};

use metal::MTLComputeCommandEncoder;

use crate::backends::metal::{
    ComputePipelineState, KernelDataType, MTLBuffer, MTLCommandBuffer,
    MTLCommandEncoder, MTLContext, MTLError, MTLSize, ProtocolObject,
};

// ---- Finalize Kernel ----

#[derive(Debug, thiserror::Error)]
pub enum MoeFinalizeError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
}

pub struct MoeFinalizeKernel {
    pipeline_f16: ComputePipelineState,
    pipeline_bf16: ComputePipelineState,
    pipeline_f32: ComputePipelineState,
}

#[derive(Debug)]
pub struct MoeFinalizeArguments<'a> {
    pub tok2row_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [T*K] i32
    pub probs_buffer: &'a ProtocolObject<dyn MTLBuffer>,   // [T*K] f16/bf16
    pub y_partial_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [sum_k, d_model] f16
    pub y_out_buffer: &'a ProtocolObject<dyn MTLBuffer>,     // [T, d_model] f16
    pub t: usize,
    pub d_model: usize,
    pub k: usize,
}

impl MoeFinalizeKernel {
    pub fn new(ctx: &MTLContext) -> Result<Self, MoeFinalizeError> {
        let pipeline_f16 =
            ctx.compute_pipeline_state("moe_finalize_f16", None)?;
        let pipeline_bf16 =
            ctx.compute_pipeline_state("moe_finalize_bf16", None)?;
        let pipeline_f32 =
            ctx.compute_pipeline_state("moe_finalize_f32", None)?;
        Ok(Self {
            pipeline_f16,
            pipeline_bf16,
            pipeline_f32,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        args: MoeFinalizeArguments,
        dtype: KernelDataType,
    ) -> Result<(), MoeFinalizeError> {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        match dtype {
            KernelDataType::Float16 => {
                encoder.set_compute_pipeline_state(&self.pipeline_f16);
            },
            KernelDataType::BFloat16 => {
                encoder.set_compute_pipeline_state(&self.pipeline_bf16);
            },
            KernelDataType::Float32 => {
                encoder.set_compute_pipeline_state(&self.pipeline_f32);
            },
        }
        encoder.set_buffer(Some(args.tok2row_buffer), 0, 0);
        encoder.set_buffer(Some(args.probs_buffer), 0, 1);
        encoder.set_buffer(Some(args.y_partial_buffer), 0, 2);
        encoder.set_buffer(Some(args.y_out_buffer), 0, 3);
        let t_u = args.t as u32;
        let dm_u = args.d_model as u32;
        let k_u = args.k as u32;
        unsafe {
            encoder.set_bytes(
                NonNull::new(&t_u as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                4,
            );
            encoder.set_bytes(
                NonNull::new(&dm_u as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                5,
            );
            encoder.set_bytes(
                NonNull::new(&k_u as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                6,
            );
        }

        // Launch tiles over (N tiles, M tiles)
        const BM: usize = 32;
        const BN: usize = 64;
        let num_tiles_n = (args.d_model + BN - 1) / BN;
        let num_tiles_m = (args.t + BM - 1) / BM;
        let threads_per_threadgroup = MTLSize::new(128, 1, 1); // BM * BN * 32 = 1 * 4 * 32
        if num_tiles_m > 0 && num_tiles_n > 0 {
            let threadgroups = MTLSize::new(num_tiles_n, num_tiles_m, 1);
            encoder
                .dispatch_threadgroups(threadgroups, threads_per_threadgroup);
        }
        encoder.end_encoding();
        Ok(())
    }
}
