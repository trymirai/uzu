// Fused simple decode: Pass A + Fused Pass B (no y_partial buffer, no finalize)

use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, CommandBufferRef,
    ComputePipelineState as MTLComputePipelineState, FunctionConstantValues,
    MTLDataType, MTLSize,
};

use super::{dtype_index, dtype_suffix};
use crate::backends::metal::{KernelDataType, MTLContext, MTLError};

#[derive(Debug, thiserror::Error)]
pub enum MoeSimpleDecodeFusedError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
}

/// Arguments for fused single-token MoE decode
#[derive(Debug)]
pub struct MoeSimpleDecodeFusedArguments<'a> {
    /// Input activation [d_model]
    pub x: &'a MTLBuffer,
    /// Top-K expert indices from router [K]
    pub topk_ids: &'a MTLBuffer,
    /// Top-K probabilities from router [K]
    pub topk_probs: &'a MTLBuffer,
    /// Up/gate projection weights [E, 2*d_ff, d_model]
    pub w13_all: &'a MTLBuffer,
    /// Down projection weights [E, d_model, d_ff]
    pub w2_all: &'a MTLBuffer,
    /// Up/gate biases [E, 2*d_ff]
    pub up_biases: &'a MTLBuffer,
    /// Down biases [E, d_model]
    pub down_biases: &'a MTLBuffer,
    /// Hidden buffer [K, d_ff] - intermediate storage (f32)
    pub hidden: &'a MTLBuffer,
    /// Final output [d_model] - NO y_partial needed!
    pub y: &'a MTLBuffer,
    /// Model dimension
    pub d_model: usize,
    /// FFN hidden dimension
    pub d_ff: usize,
    /// Number of active experts per token
    pub k: usize,
    /// Gating activation: 0=GELU, 1=SiLU, 2=SwiGLU, 3=GEGLU
    pub gating_code: u32,
    /// Data type
    pub data_type: KernelDataType,
}

pub struct MoeSimpleDecodeFusedKernel {
    pass_a: Vec<Vec<MTLComputePipelineState>>, // [gate][dtype]
    pass_b_fused: Vec<MTLComputePipelineState>, // [dtype]
}

impl MoeSimpleDecodeFusedKernel {
    pub fn new(ctx: &MTLContext) -> Result<Self, MoeSimpleDecodeFusedError> {
        let dtypes = [
            KernelDataType::Float16,
            KernelDataType::BFloat16,
            KernelDataType::Float32,
        ];

        // Pass A: needs gating function constant
        let mut pass_a = vec![Vec::with_capacity(dtypes.len()); 4];
        for gate in 0u32..4u32 {
            for dtype in &dtypes {
                let suffix = dtype_suffix(*dtype);
                let fcv = FunctionConstantValues::new();
                fcv.set_constant_value_at_index(
                    &gate as *const u32 as *const std::ffi::c_void,
                    MTLDataType::UInt,
                    30,
                );
                let kernel_name = format!("moe_simple_decode_fused_pass_a_{}", suffix);
                let cache_key = format!("{}_gate_{}", kernel_name, gate);
                pass_a[gate as usize].push(
                    ctx.compute_pipeline_state_cached(&cache_key, &kernel_name, Some(&fcv))?,
                );
            }
        }

        // Fused Pass B: no function constants needed
        let mut pass_b_fused = Vec::with_capacity(dtypes.len());
        for dtype in &dtypes {
            let suffix = dtype_suffix(*dtype);
            let kernel_name = format!("moe_simple_decode_fused_pass_b_{}", suffix);
            pass_b_fused.push(ctx.compute_pipeline_state(&kernel_name, None)?);
        }

        Ok(Self { pass_a, pass_b_fused })
    }

    /// Encode the fused decode pipeline (2 passes instead of 3)
    ///
    /// Dispatches:
    /// 1. Pass A: grid = (h_blocks, K) - K parallel expert up+gate projections
    /// 2. Fused Pass B: grid = (d_blocks, 1) - computes final y directly
    pub fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        args: MoeSimpleDecodeFusedArguments,
    ) -> Result<(), MoeSimpleDecodeFusedError> {
        if args.k == 0 {
            return Ok(());
        }

        let gate_idx = (args.gating_code.min(3)) as usize;
        let dtype_idx = dtype_index(args.data_type);

        let d_model_u32 = args.d_model as u32;
        let d_ff_u32 = args.d_ff as u32;
        let k_u32 = args.k as u32;

        // Pass A: 4 simdgroups (128 threads), each threadgroup outputs 4 elements
        const PASSA_BLOCK_H: u32 = 4;
        const PASSA_THREADS: u32 = 128;
        let h_blocks = (d_ff_u32 + PASSA_BLOCK_H - 1) / PASSA_BLOCK_H;

        // Pass A: x @ W13[expert] -> hidden
        {
            let pipeline = &self.pass_a[gate_idx][dtype_idx];
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(args.x), 0);
            encoder.set_buffer(1, Some(args.topk_ids), 0);
            encoder.set_buffer(2, Some(args.w13_all), 0);
            encoder.set_buffer(3, Some(args.up_biases), 0);
            encoder.set_buffer(4, Some(args.hidden), 0);
            encoder.set_bytes(5, size_of::<u32>() as u64, &d_model_u32 as *const u32 as *const _);
            encoder.set_bytes(6, size_of::<u32>() as u64, &d_ff_u32 as *const u32 as *const _);
            encoder.set_bytes(7, size_of::<u32>() as u64, &k_u32 as *const u32 as *const _);
            encoder.dispatch_thread_groups(
                MTLSize::new(h_blocks as u64, args.k as u64, 1),
                MTLSize::new(PASSA_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
        }

        // Fused Pass B: 8 simdgroups (256 threads), outputs final y directly
        const PASSB_BLOCK_D: u32 = 8;
        const PASSB_THREADS: u32 = 256;
        let d_blocks = (d_model_u32 + PASSB_BLOCK_D - 1) / PASSB_BLOCK_D;

        // Fused Pass B: hidden @ W2[expert] -> y (with weighted sum)
        {
            let pipeline = &self.pass_b_fused[dtype_idx];
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(args.hidden), 0);
            encoder.set_buffer(1, Some(args.topk_ids), 0);
            encoder.set_buffer(2, Some(args.topk_probs), 0);
            encoder.set_buffer(3, Some(args.w2_all), 0);
            encoder.set_buffer(4, Some(args.down_biases), 0);
            encoder.set_buffer(5, Some(args.y), 0);
            encoder.set_bytes(6, size_of::<u32>() as u64, &d_model_u32 as *const u32 as *const _);
            encoder.set_bytes(7, size_of::<u32>() as u64, &d_ff_u32 as *const u32 as *const _);
            encoder.set_bytes(8, size_of::<u32>() as u64, &k_u32 as *const u32 as *const _);
            encoder.dispatch_thread_groups(
                MTLSize::new(d_blocks as u64, 1, 1), // NOT per K!
                MTLSize::new(PASSB_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
        }

        Ok(())
    }
}
