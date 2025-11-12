use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use super::{SSMKernelError, fn_suffix};
use crate::backends::metal::{KernelDataType, MTLContext};

pub struct SSMUpdateKernel {
    pipeline: MTLComputePipelineState,
}

pub struct SSMUpdateArguments<'a> {
    /// X – Silu-activated hidden stream (per head/channel) coming from the conv.
    pub x: &'a MTLBuffer, // buffer(0)
    /// Δt – learned step sizes per head.
    pub dt: &'a MTLBuffer, // buffer(1)
    /// A – diagonal decay coefficients shared across tokens (eigenvalues).
    pub a: &'a MTLBuffer, // buffer(2)
    /// B – input projection that injects X into the state space.
    pub b: &'a MTLBuffer, // buffer(3)
    /// C – output projection that maps state to residual updates.
    pub c: &'a MTLBuffer, // buffer(4)
    /// D – per-head skip/identity weights.
    pub d: &'a MTLBuffer, // buffer(5)
    /// Z – post-gate activations that modulate the residual output.
    pub z: &'a MTLBuffer, // buffer(6)
    /// Current state tensor.
    pub state: &'a MTLBuffer, // buffer(7)
    /// Y – residual output buffer.
    pub y: &'a MTLBuffer, // buffer(8)
    /// Next state tensor (same buffer as `state`, but written after update).
    pub next_state: &'a MTLBuffer, // buffer(9)
    pub batch_size: usize,
    pub channels: usize,
}

impl SSMUpdateKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, SSMKernelError> {
        let fn_name = format!("ssm_update_kernel_{}", fn_suffix(data_type));
        let pipeline = context
            .compute_pipeline_state(&fn_name, None)
            .map_err(SSMKernelError::MetalError)?;
        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: SSMUpdateArguments,
    ) -> Result<(), SSMKernelError> {
        compute_encoder.set_compute_pipeline_state(&self.pipeline);

        compute_encoder.set_buffer(0, Some(args.x), 0);
        compute_encoder.set_buffer(1, Some(args.dt), 0);
        compute_encoder.set_buffer(2, Some(args.a), 0);
        compute_encoder.set_buffer(3, Some(args.b), 0);
        compute_encoder.set_buffer(4, Some(args.c), 0);
        compute_encoder.set_buffer(5, Some(args.d), 0);
        compute_encoder.set_buffer(6, Some(args.z), 0);
        compute_encoder.set_buffer(7, Some(args.state), 0);
        compute_encoder.set_buffer(8, Some(args.y), 0);
        compute_encoder.set_buffer(9, Some(args.next_state), 0);

        let threads_per_threadgroup = MTLSize {
            width: 32,
            height: 32,
            depth: 1,
        };
        let total_threads = MTLSize {
            width: args.batch_size as u64,
            height: args.channels as u64,
            depth: 1,
        };
        compute_encoder
            .dispatch_threads(total_threads, threads_per_threadgroup);
        Ok(())
    }
}
