use std::{cell::RefCell, mem::size_of};

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, Device as MTLDevice,
    MTLResourceOptions, MTLSize,
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

// ---- Bucket Counts Kernel ----

#[derive(Debug, thiserror::Error)]
pub enum MoeBucketCountsError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("Invalid dimensions: T={t}, E={e}, K={k}")]
    InvalidDimensions {
        t: usize,
        e: usize,
        k: usize,
    },
}

pub struct MoeBucketCountsKernel {
    pipeline_partials: MTLComputePipelineState,
    pipeline_reduce: MTLComputePipelineState,
    device: MTLDevice,
}

#[derive(Debug)]
pub struct MoeBucketCountsArguments<'a> {
    pub topk_ids_buffer: &'a MTLBuffer,
    pub counts_buffer: &'a MTLBuffer,
    pub t: usize,
    pub e: usize,
    pub k: usize,
}

// ---- Offsets Scan Kernel ----

#[derive(Debug, thiserror::Error)]
pub enum MoeOffsetsScanError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
}

pub struct MoeOffsetsScanKernel {
    pipeline: MTLComputePipelineState,
}

#[derive(Debug)]
pub struct MoeOffsetsScanArguments<'a> {
    pub counts_buffer: &'a MTLBuffer,  // [E]
    pub offsets_buffer: &'a MTLBuffer, // [E+1]
    pub sumk_buffer: &'a MTLBuffer,    // [1]
    pub e: usize,
}

impl MoeOffsetsScanKernel {
    pub fn new(mtl_context: &MTLContext) -> Result<Self, MoeOffsetsScanError> {
        let pipeline = mtl_context
            .compute_pipeline_state("moe_offsets_exclusive_scan", None)?;
        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: MoeOffsetsScanArguments,
    ) -> Result<(), MoeOffsetsScanError> {
        compute_encoder.set_compute_pipeline_state(&self.pipeline);
        compute_encoder.set_buffer(0, Some(args.counts_buffer), 0);
        compute_encoder.set_buffer(1, Some(args.offsets_buffer), 0);
        compute_encoder.set_buffer(2, Some(args.sumk_buffer), 0);
        let e_u32 = args.e as u32;
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const std::ffi::c_void,
        );

        // Single threadgroup implementation (BLOCK_SIZE=256), repeated in-kernel
        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let threadgroups = MTLSize::new(1, 1, 1);
        compute_encoder
            .dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        Ok(())
    }
}

// ---- Scatter Buckets Kernels ----

#[derive(Debug, thiserror::Error)]
pub enum MoeScatterError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
}

pub struct MoeScatterKernels {
    pipeline_bases: MTLComputePipelineState,
    pipeline_scatter_f16: MTLComputePipelineState,
    pipeline_scatter_f32: MTLComputePipelineState,
    pipeline_scatter_bf16: MTLComputePipelineState,
    device: MTLDevice,
    last_block_alloc: RefCell<Option<MTLBuffer>>,
}

#[derive(Debug)]
pub struct MoeBlockBasesArguments<'a> {
    pub partials_buffer: &'a MTLBuffer, // [num_blocks * num_tiles * 512]
    pub block_bases_buffer: &'a MTLBuffer, // same shape as partials
    pub e: usize,
    pub num_blocks: usize,
    pub num_tiles: usize,
}

#[derive(Debug)]
pub struct MoeScatterArguments<'a> {
    pub topk_ids_buffer: &'a MTLBuffer,
    pub topk_probs_buffer: &'a MTLBuffer,
    pub offsets_buffer: &'a MTLBuffer,
    pub block_bases_buffer: &'a MTLBuffer,
    pub out_ids_buffer: &'a MTLBuffer,
    pub out_probs_buffer: &'a MTLBuffer,
    pub t: usize,
    pub e: usize,
    pub k: usize,
    pub num_blocks: usize,
    pub num_tiles: usize,
}

impl MoeScatterKernels {
    pub fn new(mtl_context: &MTLContext) -> Result<Self, MoeScatterError> {
        let pipeline_bases = mtl_context
            .compute_pipeline_state("moe_block_bases_from_partials", None)?;
        let pipeline_scatter_f16 = mtl_context
            .compute_pipeline_state("moe_scatter_buckets_f16", None)?;
        let pipeline_scatter_f32 = mtl_context
            .compute_pipeline_state("moe_scatter_buckets_f32", None)?;
        let pipeline_scatter_bf16 = mtl_context
            .compute_pipeline_state("moe_scatter_buckets_bf16", None)?;
        Ok(Self {
            pipeline_bases,
            pipeline_scatter_f16,
            pipeline_scatter_f32,
            pipeline_scatter_bf16,
            device: mtl_context.device.clone(),
            last_block_alloc: RefCell::new(None),
        })
    }

    pub fn encode_block_bases(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: MoeBlockBasesArguments,
    ) -> Result<(), MoeScatterError> {
        compute_encoder.set_compute_pipeline_state(&self.pipeline_bases);
        compute_encoder.set_buffer(0, Some(args.partials_buffer), 0);
        compute_encoder.set_buffer(1, Some(args.block_bases_buffer), 0);
        // Allocate block_alloc buffer and keep it for scatter
        let entries = args.num_blocks * args.num_tiles * 512usize;
        let block_alloc = self.device.new_buffer(
            (entries * size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        compute_encoder.set_buffer(2, Some(&block_alloc), 0);
        let e_u32 = args.e as u32;
        let nb_u32 = args.num_blocks as u32;
        let nt_u32 = args.num_tiles as u32;
        let cap_u32: u32 = 0;
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &nb_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &nt_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &cap_u32 as *const u32 as *const std::ffi::c_void,
        );

        let total_entries = args.num_tiles * 512usize;
        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let tg = MTLSize::new(((total_entries + 255) / 256) as u64, 1, 1);
        if total_entries > 0 {
            compute_encoder.dispatch_thread_groups(tg, threads_per_threadgroup);
        }
        // Stash for subsequent scatter call
        self.last_block_alloc.borrow_mut().replace(block_alloc);
        Ok(())
    }

    pub fn encode_scatter(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: MoeScatterArguments,
    ) -> Result<(), MoeScatterError> {
        // For now default to f16 probabilities
        compute_encoder.set_compute_pipeline_state(&self.pipeline_scatter_f16);
        compute_encoder.set_buffer(0, Some(args.topk_ids_buffer), 0);
        compute_encoder.set_buffer(1, Some(args.topk_probs_buffer), 0);
        compute_encoder.set_buffer(2, Some(args.offsets_buffer), 0);
        compute_encoder.set_buffer(3, Some(args.block_bases_buffer), 0);
        // Use the last computed block_alloc
        let block_alloc_opt = self.last_block_alloc.borrow();
        let block_alloc_buf = block_alloc_opt.as_ref().ok_or_else(|| {
            MoeScatterError::MetalError(MTLError::Generic(
                "block_alloc buffer missing; call encode_block_bases first"
                    .to_string(),
            ))
        })?;
        compute_encoder.set_buffer(4, Some(block_alloc_buf), 0);
        compute_encoder.set_buffer(5, Some(args.out_ids_buffer), 0);
        compute_encoder.set_buffer(6, Some(args.out_probs_buffer), 0);
        let t_u32 = args.t as u32;
        let e_u32 = args.e as u32;
        let k_u32 = args.k as u32;
        let nb_u32 = args.num_blocks as u32;
        let nt_u32 = args.num_tiles as u32;
        compute_encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            &t_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            9,
            size_of::<u32>() as u64,
            &k_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            10,
            size_of::<u32>() as u64,
            &nb_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            11,
            size_of::<u32>() as u64,
            &nt_u32 as *const u32 as *const std::ffi::c_void,
        );

        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let tg = MTLSize::new(args.num_blocks as u64, 1, 1);
        if args.num_blocks > 0 {
            compute_encoder.dispatch_thread_groups(tg, threads_per_threadgroup);
        }
        Ok(())
    }
}
impl MoeBucketCountsKernel {
    pub fn new(mtl_context: &MTLContext) -> Result<Self, MoeBucketCountsError> {
        let pipeline_partials =
            mtl_context.compute_pipeline_state("moe_bucket_partials", None)?;
        let pipeline_reduce = mtl_context
            .compute_pipeline_state("moe_bucket_reduce_partials", None)?;
        Ok(Self {
            pipeline_partials,
            pipeline_reduce,
            device: mtl_context.device.clone(),
        })
    }

    pub fn encode(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: MoeBucketCountsArguments,
    ) -> Result<(), MoeBucketCountsError> {
        if args.k == 0 || args.e == 0 {
            return Ok(());
        }
        if args.t == 0 {
            return Ok(());
        }
        if args.e < 1 {
            return Err(MoeBucketCountsError::InvalidDimensions {
                t: args.t,
                e: args.e,
                k: args.k,
            });
        }

        // Pass A: partials
        compute_encoder.set_compute_pipeline_state(&self.pipeline_partials);
        compute_encoder.set_buffer(0, Some(args.topk_ids_buffer), 0);

        // Compute sizes for partials buffer
        let num_blocks = ((args.t + 255) / 256) as u32;
        let num_tiles = (args.e as u32 + 512 - 1) / 512;
        let partials_len =
            (num_blocks as usize) * (num_tiles as usize) * 512usize;
        let partials_size = (partials_len * size_of::<u32>()) as u64;
        let partials_buf = self
            .device
            .new_buffer(partials_size, MTLResourceOptions::StorageModeShared);

        compute_encoder.set_buffer(1, Some(&partials_buf), 0);
        let t_u32 = args.t as u32;
        let e_u32 = args.e as u32;
        let k_u32 = args.k as u32;
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &t_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &k_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &num_blocks as *const u32 as *const std::ffi::c_void,
        );

        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        if num_blocks > 0 {
            let tg = MTLSize::new(num_blocks as u64, 1, 1);
            compute_encoder.dispatch_thread_groups(tg, threads_per_threadgroup);
        }

        // Pass B: reduce
        compute_encoder.set_compute_pipeline_state(&self.pipeline_reduce);
        compute_encoder.set_buffer(0, Some(&partials_buf), 0);
        compute_encoder.set_buffer(1, Some(args.counts_buffer), 0);
        compute_encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const std::ffi::c_void,
        );
        compute_encoder.set_bytes(
            3,
            size_of::<u32>() as u64,
            &num_blocks as *const u32 as *const std::ffi::c_void,
        );
        let tg2 = MTLSize::new(((args.e + 255) / 256) as u64, 1, 1);
        if args.e > 0 {
            compute_encoder
                .dispatch_thread_groups(tg2, threads_per_threadgroup);
        }
        Ok(())
    }
}

impl MoeTopKKernel {
    pub fn new(mtl_context: &MTLContext) -> Result<Self, MoeTopKError> {
        let pipeline_f16 =
            mtl_context.compute_pipeline_state("moe_topk_select_f16", None)?;
        let pipeline_f32 =
            mtl_context.compute_pipeline_state("moe_topk_select_f32", None)?;
        Ok(Self {
            pipeline_f16,
            pipeline_f32,
        })
    }

    pub fn encode(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
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

        match dtype {
            KernelDataType::Float16 => {
                compute_encoder.set_compute_pipeline_state(&self.pipeline_f16)
            },
            KernelDataType::Float32 => {
                compute_encoder.set_compute_pipeline_state(&self.pipeline_f32)
            },
            KernelDataType::BFloat16 => {
                // Not supported in v1
                compute_encoder.set_compute_pipeline_state(&self.pipeline_f32)
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

        Ok(())
    }
}
