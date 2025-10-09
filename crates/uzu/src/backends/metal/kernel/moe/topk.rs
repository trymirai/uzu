use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, CommandBufferRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use crate::backends::metal::{KernelDataType, MTLContext, MTLError};

#[derive(Debug, thiserror::Error)]
pub enum MoeTopKError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("Invalid dimensions: T={t}, E={e}, K={k}")]
    InvalidDimensions {
        t: usize,
        e: usize,
        k: usize,
    },
}

pub struct MoeTopKKernel {
    pipeline_f16: MTLComputePipelineState,
    pipeline_f32: MTLComputePipelineState,
    pipeline_bf16: MTLComputePipelineState,
}

#[derive(Debug)]
pub struct MoeTopKArguments<'a> {
    pub logits_buffer: &'a MTLBuffer,
    pub topk_ids_buffer: &'a MTLBuffer,
    pub topk_probs_buffer: &'a MTLBuffer,
    pub t: usize,
    pub e: usize,
    pub k: usize,
    pub renorm: bool,
}

/// Encode MoE TopK kernel (standalone, reusable)
pub fn encode_moe_topk(
    ctx: &MTLContext,
    command_buffer: &CommandBufferRef,
    dtype: KernelDataType,
    args: &MoeTopKArguments,
) -> Result<(), MTLError> {
    let kernel_name = match dtype {
        KernelDataType::BFloat16 => "moe_topk_select_bf16",
        KernelDataType::Float16 => "moe_topk_select_f16",
        KernelDataType::Float32 => "moe_topk_select_f32",
    };

    let pipeline = ctx.compute_pipeline_state(kernel_name, None)?;
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(args.logits_buffer), 0);
    encoder.set_buffer(1, Some(args.topk_ids_buffer), 0);
    encoder.set_buffer(2, Some(args.topk_probs_buffer), 0);

    let t_u32 = args.t as u32;
    let e_u32 = args.e as u32;
    let k_u32 = args.k as u32;
    let renorm_u32 = if args.renorm {
        1u32
    } else {
        0u32
    };
    encoder.set_bytes(
        3,
        size_of::<u32>() as u64,
        &t_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        4,
        size_of::<u32>() as u64,
        &e_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        5,
        size_of::<u32>() as u64,
        &k_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        6,
        size_of::<u32>() as u64,
        &renorm_u32 as *const u32 as *const _,
    );

    encoder.dispatch_thread_groups(
        MTLSize::new(args.t as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.end_encoding();
    Ok(())
}

/// Encode MoE TopK using a prebuilt pipeline (avoids pipeline lookup)
pub fn encode_moe_topk_with_pipeline(
    pipeline: &MTLComputePipelineState,
    command_buffer: &CommandBufferRef,
    args: &MoeTopKArguments,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(args.logits_buffer), 0);
    encoder.set_buffer(1, Some(args.topk_ids_buffer), 0);
    encoder.set_buffer(2, Some(args.topk_probs_buffer), 0);

    let t_u32 = args.t as u32;
    let e_u32 = args.e as u32;
    let k_u32 = args.k as u32;
    let renorm_u32 = if args.renorm {
        1u32
    } else {
        0u32
    };
    encoder.set_bytes(
        3,
        size_of::<u32>() as u64,
        &t_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        4,
        size_of::<u32>() as u64,
        &e_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        5,
        size_of::<u32>() as u64,
        &k_u32 as *const u32 as *const _,
    );
    encoder.set_bytes(
        6,
        size_of::<u32>() as u64,
        &renorm_u32 as *const u32 as *const _,
    );

    encoder.dispatch_thread_groups(
        MTLSize::new(args.t as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.end_encoding();
}

impl MoeTopKKernel {
    pub fn new(mtl_context: &MTLContext) -> Result<Self, MoeTopKError> {
        let pipeline_f16 =
            mtl_context.compute_pipeline_state("moe_topk_select_f16", None)?;
        let pipeline_f32 =
            mtl_context.compute_pipeline_state("moe_topk_select_f32", None)?;
        let pipeline_bf16 =
            mtl_context.compute_pipeline_state("moe_topk_select_bf16", None)?;
        Ok(Self {
            pipeline_f16,
            pipeline_f32,
            pipeline_bf16,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        dtype: KernelDataType,
        args: MoeTopKArguments,
    ) -> Result<(), MoeTopKError> {
        if args.k == 0 || args.e == 0 || args.t == 0 {
            // No-op for empty work; allow t==0 silently
            if args.t == 0 {
                return Ok(());
            }
        }
        if args.e < args.k {
            return Err(MoeTopKError::InvalidDimensions {
                t: args.t,
                e: args.e,
                k: args.k,
            });
        }

        let compute_encoder = command_buffer.new_compute_command_encoder();
        match dtype {
            KernelDataType::Float16 => {
                compute_encoder.set_compute_pipeline_state(&self.pipeline_f16)
            },
            KernelDataType::Float32 => {
                compute_encoder.set_compute_pipeline_state(&self.pipeline_f32)
            },
            KernelDataType::BFloat16 => {
                compute_encoder.set_compute_pipeline_state(&self.pipeline_bf16)
            },
        }

        compute_encoder.set_buffer(0, Some(args.logits_buffer), 0);
        compute_encoder.set_buffer(1, Some(args.topk_ids_buffer), 0);
        compute_encoder.set_buffer(2, Some(args.topk_probs_buffer), 0);

        let t_u32 = args.t as u32;
        let e_u32 = args.e as u32;
        let k_u32 = args.k as u32;
        let renorm_u32: u32 = if args.renorm {
            1
        } else {
            0
        };

        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &t_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &k_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &renorm_u32 as *const u32 as *const std::ffi::c_void,
        );

        // Launch
        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let num_threadgroups = if args.t == 0 {
            0
        } else {
            (args.t + 255) / 256
        } as u64;
        if num_threadgroups > 0 {
            let threadgroups = MTLSize::new(num_threadgroups, 1, 1);
            compute_encoder
                .dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        }
        compute_encoder.end_encoding();

        Ok(())
    }
}
