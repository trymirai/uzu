use std::{ffi::c_void, mem::size_of, ptr::NonNull};

use metal::MTLComputeCommandEncoder;

use crate::backends::metal::{
    KernelDataType, MTLBuffer, MTLCommandBuffer,
    MTLCommandEncoder, MTLComputePipelineState, MTLContext, MTLError, MTLSize, ProtocolObject, Retained,
};

const THREADS_PER_THREADGROUP: usize = 256;
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

pub struct MoeRouterTopKKernel {
    pipeline_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_f32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_bf16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
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

impl MoeRouterTopKKernel {
    pub fn new(mtl_context: &MTLContext) -> Result<Self, MoeRouterTopKError> {
        let pipeline_f16 =
            mtl_context.compute_pipeline_state("moe_router_topk_f16", None)?;
        let pipeline_f32 =
            mtl_context.compute_pipeline_state("moe_router_topk_f32", None)?;
        let pipeline_bf16 =
            mtl_context.compute_pipeline_state("moe_router_topk_bf16", None)?;
        Ok(Self {
            pipeline_f16,
            pipeline_f32,
            pipeline_bf16,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        dtype: KernelDataType,
        args: MoeRouterTopKArguments,
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

        let compute_encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
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

        compute_encoder.set_buffer(Some(args.input_buffer), 0, 0);
        compute_encoder.set_buffer(Some(args.weight_buffer), 0, 1);
        compute_encoder.set_buffer(Some(args.bias_buffer), 0, 2);
        compute_encoder.set_buffer(Some(args.topk_ids_buffer), 0, 3);
        compute_encoder.set_buffer(Some(args.topk_probs_buffer), 0, 4);

        let t_u32 = args.t as u32;
        let d_u32 = args.d_model as u32;
        let e_u32 = args.e as u32;
        let k_u32 = args.k as u32;
        let renorm_u32: u32 = if args.renorm {
            1
        } else {
            0
        };

        unsafe {
            compute_encoder.set_bytes(
                NonNull::new(&t_u32 as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                5,
            );
            compute_encoder.set_bytes(
                NonNull::new(&d_u32 as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                6,
            );
            compute_encoder.set_bytes(
                NonNull::new(&e_u32 as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                7,
            );
            compute_encoder.set_bytes(
                NonNull::new(&k_u32 as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                8,
            );
            compute_encoder.set_bytes(
                NonNull::new(&renorm_u32 as *const u32 as *mut c_void).unwrap(),
                size_of::<u32>(),
                9,
            );
        }

        let threadgroups = MTLSize::new(1, args.t, 1);
        let threads_per_tg = MTLSize::new(THREADS_PER_THREADGROUP, 1, 1);

        compute_encoder.dispatch_threadgroups(threadgroups, threads_per_tg);
        compute_encoder.end_encoding();

        Ok(())
    }
}
