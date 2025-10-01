use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState,
    FunctionConstantValues, MTLDataType, MTLResourceOptions, MTLSize,
};

use crate::backends::metal::{KernelDataType, MTLContext, MTLError};

mod encodable;
pub use encodable::{MoeBlockEncodable, SharedMoeWeights};

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
}

#[derive(Debug)]
pub struct MoeBucketCountsArguments<'a> {
    pub topk_ids_buffer: &'a MTLBuffer,
    pub counts_buffer: &'a MTLBuffer,
    pub partials_buffer: &'a MTLBuffer,
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
#[derive(Debug, thiserror::Error)]
pub enum MoeExpertsError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
}

pub struct MoeExpertsKernel {
    pipelines_f16: Vec<MTLComputePipelineState>, // 0=GELU,1=SiLU,2=SwiGLU,3=GEGLU
    tile_counts_pipeline: MTLComputePipelineState,
    tile_scan_pipeline: MTLComputePipelineState,
    tile_build_map_pipeline: MTLComputePipelineState,
}

#[derive(Debug)]
pub struct MoeExpertsArguments<'a> {
    pub x_buffer: &'a MTLBuffer,           // [T,d_model]
    pub bucketed_token_ids: &'a MTLBuffer, // [sum_k]
    pub expert_offsets: &'a MTLBuffer,     // [E+1]
    pub w1_all: &'a MTLBuffer,             // [E*d_ff*d_model]
    pub w3_all: &'a MTLBuffer,             // [E*d_ff*d_model] or empty
    pub w2_all: &'a MTLBuffer,             // [E*d_model*d_ff]
    pub y_partial: &'a MTLBuffer,          // [sum_k,d_model]
    pub up_biases: &'a MTLBuffer, // [E*2*d_ff] if fused, [E*d_ff] otherwise
    pub down_biases: &'a MTLBuffer, // [E*d_model]
    pub t: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub e: usize,
    pub k: usize, // num_experts_per_token
    pub gating_code: u32, // 0=GELU,1=SiLU,2=SwiGLU,3=GEGLU
    pub gate_clip_min: f32,
    pub gate_clip_max: f32,
    pub up_clip_min: f32,
    pub up_clip_max: f32,
    pub silu_alpha: f32,
}

impl MoeExpertsKernel {
    pub fn new(ctx: &MTLContext) -> Result<Self, MoeExpertsError> {
        let mut pipelines_f16 = Vec::with_capacity(4);
        for gate in 0u32..4u32 {
            let fcv = FunctionConstantValues::new();
            fcv.set_constant_value_at_index(
                &gate as *const u32 as *const std::ffi::c_void,
                MTLDataType::UInt,
                30,
            );
            let debug_mask: u32 = 0;
            fcv.set_constant_value_at_index(
                &debug_mask as *const u32 as *const std::ffi::c_void,
                MTLDataType::UInt,
                31,
            );
            let pipeline = ctx.compute_pipeline_state(
                "moe_fused_expert_mlp_f16",
                Some(&fcv),
            )?;
            pipelines_f16.push(pipeline);
        }
        let tile_counts = ctx.compute_pipeline_state("moe_tile_counts", None)?;
        let tile_scan = ctx.compute_pipeline_state("moe_tile_scan", None)?;
        let tile_build = ctx.compute_pipeline_state("moe_build_tile_map", None)?;

        Ok(Self {
            pipelines_f16,
            tile_counts_pipeline: tile_counts,
            tile_scan_pipeline: tile_scan,
            tile_build_map_pipeline: tile_build,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        args: MoeExpertsArguments,
    ) -> Result<(), MoeExpertsError> {
        // Select specialized pipeline by gating code
        let gate_idx = (args.gating_code.min(3)) as usize;
        encoder.set_compute_pipeline_state(&self.pipelines_f16[gate_idx]);
        encoder.set_buffer(0, Some(args.x_buffer), 0);
        encoder.set_buffer(1, Some(args.bucketed_token_ids), 0);
        encoder.set_buffer(2, Some(args.expert_offsets), 0);
        encoder.set_buffer(3, Some(args.w1_all), 0);
        encoder.set_buffer(4, Some(args.w3_all), 0);
        encoder.set_buffer(5, Some(args.w2_all), 0);
        encoder.set_buffer(6, Some(args.y_partial), 0);
        let t_u = args.t as u32;
        let dm_u = args.d_model as u32;
        let dff_u = args.d_ff as u32;
        let e_u = args.e as u32;
        let gate = args.gating_code as u32;
        encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            &t_u as *const u32 as *const _,
        );
        encoder.set_bytes(
            8,
            size_of::<u32>() as u64,
            &dm_u as *const u32 as *const _,
        );
        encoder.set_bytes(
            9,
            size_of::<u32>() as u64,
            &dff_u as *const u32 as *const _,
        );
        encoder.set_bytes(
            10,
            size_of::<u32>() as u64,
            &e_u as *const u32 as *const _,
        );
        encoder.set_bytes(
            11,
            size_of::<u32>() as u64,
            &gate as *const u32 as *const _,
        );
        encoder.set_buffer(12, Some(args.up_biases), 0);
        encoder.set_buffer(13, Some(args.down_biases), 0);
        encoder.set_bytes(14, std::mem::size_of::<f32>() as u64, &args.gate_clip_min as *const f32 as *const _);
        encoder.set_bytes(15, std::mem::size_of::<f32>() as u64, &args.gate_clip_max as *const f32 as *const _);
        encoder.set_bytes(16, std::mem::size_of::<f32>() as u64, &args.up_clip_min as *const f32 as *const _);
        encoder.set_bytes(17, std::mem::size_of::<f32>() as u64, &args.up_clip_max as *const f32 as *const _);
        encoder.set_bytes(18, std::mem::size_of::<f32>() as u64, &args.silu_alpha as *const f32 as *const _);
        
        const BN: usize = 64;

        // Build compact tile map on GPU: counts -> scan
        // Get device from a buffer we already have
        let device = args.expert_offsets.device();
        let tile_counts_buf = device.new_buffer((args.e * size_of::<u32>()) as u64, MTLResourceOptions::StorageModeShared);
        let tile_offsets_buf = device.new_buffer(((args.e + 1) * size_of::<u32>()) as u64, MTLResourceOptions::StorageModeShared);
        let total_tiles_buf = device.new_buffer(size_of::<u32>() as u64, MTLResourceOptions::StorageModeShared);

        // Pass A: tile counts
        encoder.set_compute_pipeline_state(&self.tile_counts_pipeline);
        encoder.set_buffer(0, Some(args.expert_offsets), 0);
        encoder.set_buffer(1, Some(&tile_counts_buf), 0);
        let e_u32 = args.e as u32;
        encoder.set_bytes(2, size_of::<u32>() as u64, &e_u32 as *const u32 as *const _);
        encoder.dispatch_thread_groups(
            MTLSize::new(((args.e + 255) / 256) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );

        // Pass B: scan (single TG)
        encoder.set_compute_pipeline_state(&self.tile_scan_pipeline);
        encoder.set_buffer(0, Some(&tile_counts_buf), 0);
        encoder.set_buffer(1, Some(&tile_offsets_buf), 0);
        encoder.set_buffer(2, Some(&total_tiles_buf), 0);
        encoder.set_bytes(3, size_of::<u32>() as u64, &e_u32 as *const u32 as *const _);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1024, 1, 1));

        // Pass C: build tile map (unused for now, will be used for indirect dispatch later)
        let total_tiles = unsafe { *(total_tiles_buf.contents() as *const u32) } as usize;
        encoder.set_compute_pipeline_state(&self.tile_build_map_pipeline);
        encoder.set_buffer(0, Some(args.expert_offsets), 0);
        encoder.set_buffer(1, Some(&tile_offsets_buf), 0);
        encoder.set_buffer(2, Some(&tile_counts_buf), 0);
        encoder.set_buffer(3, Some(&device.new_buffer(1, MTLResourceOptions::StorageModeShared)), 0);
        encoder.set_bytes(4, size_of::<u32>() as u64, &e_u32 as *const u32 as *const _);
        encoder.dispatch_thread_groups(
            MTLSize::new(((args.e + 255) / 256) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );

        // Now run experts kernel over compact tiles: grid = (num_tiles_n, total_tiles)
        let num_tiles_n = (args.d_model + BN - 1) / BN;
        let threads_per_threadgroup = MTLSize::new(128, 1, 1);
        if total_tiles > 0 && num_tiles_n > 0 {
            let threadgroups = MTLSize::new(num_tiles_n as u64, total_tiles as u64, 1);
            // Bind tile_map as buffer 19..21 inside kernel via encoded args
            encoder.set_compute_pipeline_state(&self.pipelines_f16[(args.gating_code.min(3)) as usize]);
            encoder.set_buffer(0, Some(args.x_buffer), 0);
            encoder.set_buffer(1, Some(args.bucketed_token_ids), 0);
            encoder.set_buffer(2, Some(args.expert_offsets), 0);
            encoder.set_buffer(3, Some(args.w1_all), 0);
            encoder.set_buffer(4, Some(args.w3_all), 0);
            encoder.set_buffer(5, Some(args.w2_all), 0);
            encoder.set_buffer(6, Some(args.y_partial), 0);
            let t_u = args.t as u32;
            let dm_u = args.d_model as u32;
            let dff_u = args.d_ff as u32;
            let gate = args.gating_code as u32;
            encoder.set_bytes(7, size_of::<u32>() as u64, &t_u as *const u32 as *const _);
            encoder.set_bytes(8, size_of::<u32>() as u64, &dm_u as *const u32 as *const _);
            encoder.set_bytes(9, size_of::<u32>() as u64, &dff_u as *const u32 as *const _);
            encoder.set_bytes(10, size_of::<u32>() as u64, &e_u32 as *const u32 as *const _);
            encoder.set_bytes(11, size_of::<u32>() as u64, &gate as *const u32 as *const _);
            encoder.set_buffer(12, Some(args.up_biases), 0);
            encoder.set_buffer(13, Some(args.down_biases), 0);
            encoder.set_bytes(14, size_of::<f32>() as u64, &args.gate_clip_min as *const f32 as *const _);
            encoder.set_bytes(15, size_of::<f32>() as u64, &args.gate_clip_max as *const f32 as *const _);
            encoder.set_bytes(16, size_of::<f32>() as u64, &args.up_clip_min as *const f32 as *const _);
            encoder.set_bytes(17, size_of::<f32>() as u64, &args.up_clip_max as *const f32 as *const _);
            encoder.set_bytes(18, size_of::<f32>() as u64, &args.silu_alpha as *const f32 as *const _);
            encoder.set_buffer(19, Some(&tile_offsets_buf), 0);
            encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        }
        Ok(())
    }
}

pub struct MoeScatterKernels {
    pipeline_bases: MTLComputePipelineState,
    pipeline_scatter_f16: MTLComputePipelineState,
    pipeline_scatter_f32: MTLComputePipelineState,
    pipeline_scatter_bf16: MTLComputePipelineState,
    // map variants
    pipeline_scatter_map_f16: MTLComputePipelineState,
    pipeline_scatter_map_f32: MTLComputePipelineState,
    pipeline_scatter_map_bf16: MTLComputePipelineState,
}

#[derive(Debug)]
pub struct MoeBlockBasesArguments<'a> {
    pub partials_buffer: &'a MTLBuffer, // [num_blocks * num_tiles * 512]
    pub block_bases_buffer: &'a MTLBuffer, // same shape as partials
    pub block_alloc_buffer: &'a MTLBuffer, // [num_blocks * num_tiles * 512]
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
    pub block_alloc_buffer: &'a MTLBuffer,
    pub out_ids_buffer: &'a MTLBuffer,
    pub out_probs_buffer: &'a MTLBuffer,
    pub t: usize,
    pub e: usize,
    pub k: usize,
    pub num_blocks: usize,
    pub num_tiles: usize,
}

#[derive(Debug)]
pub struct MoeScatterWithMapArguments<'a> {
    pub base: MoeScatterArguments<'a>,
    pub tok2row_buffer: &'a MTLBuffer, // [T*K] int32, initialized to -1
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
        let pipeline_scatter_map_f16 = mtl_context
            .compute_pipeline_state("moe_scatter_buckets_map_f16", None)?;
        let pipeline_scatter_map_f32 = mtl_context
            .compute_pipeline_state("moe_scatter_buckets_map_f32", None)?;
        let pipeline_scatter_map_bf16 = mtl_context
            .compute_pipeline_state("moe_scatter_buckets_map_bf16", None)?;
        
        Ok(Self {
            pipeline_bases,
            pipeline_scatter_f16,
            pipeline_scatter_f32,
            pipeline_scatter_bf16,
            pipeline_scatter_map_f16,
            pipeline_scatter_map_f32,
            pipeline_scatter_map_bf16,
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
        compute_encoder.set_buffer(2, Some(args.block_alloc_buffer), 0);
        
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
        Ok(())
    }

    pub fn encode_scatter(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: MoeScatterArguments,
        dtype: KernelDataType,
    ) -> Result<(), MoeScatterError> {
        // Select pipeline based on dtype
        match dtype {
            KernelDataType::Float16 => {
                compute_encoder
                    .set_compute_pipeline_state(&self.pipeline_scatter_f16);
            },
            KernelDataType::Float32 => {
                compute_encoder
                    .set_compute_pipeline_state(&self.pipeline_scatter_f32);
            },
            KernelDataType::BFloat16 => {
                compute_encoder
                    .set_compute_pipeline_state(&self.pipeline_scatter_bf16);
            },
        }
        compute_encoder.set_buffer(0, Some(args.topk_ids_buffer), 0);
        compute_encoder.set_buffer(1, Some(args.topk_probs_buffer), 0);
        compute_encoder.set_buffer(2, Some(args.offsets_buffer), 0);
        compute_encoder.set_buffer(3, Some(args.block_bases_buffer), 0);
        compute_encoder.set_buffer(4, Some(args.block_alloc_buffer), 0);
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

    pub fn encode_scatter_with_map(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        args: MoeScatterWithMapArguments,
        dtype: KernelDataType,
    ) -> Result<(), MoeScatterError> {
        let pipeline = match dtype {
            KernelDataType::Float16 => &self.pipeline_scatter_map_f16,
            KernelDataType::Float32 => &self.pipeline_scatter_map_f32,
            KernelDataType::BFloat16 => &self.pipeline_scatter_map_bf16,
        };
        compute_encoder.set_compute_pipeline_state(pipeline);
        let base = &args.base;
        compute_encoder.set_buffer(0, Some(base.topk_ids_buffer), 0);
        compute_encoder.set_buffer(1, Some(base.topk_probs_buffer), 0);
        compute_encoder.set_buffer(2, Some(base.offsets_buffer), 0);
        compute_encoder.set_buffer(3, Some(base.block_bases_buffer), 0);
        compute_encoder.set_buffer(4, Some(base.block_alloc_buffer), 0);
        compute_encoder.set_buffer(5, Some(base.out_ids_buffer), 0);
        compute_encoder.set_buffer(6, Some(base.out_probs_buffer), 0);
        let t_u32 = base.t as u32;
        let e_u32 = base.e as u32;
        let k_u32 = base.k as u32;
        let nb_u32 = base.num_blocks as u32;
        let nt_u32 = base.num_tiles as u32;
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
        compute_encoder.set_buffer(12, Some(args.tok2row_buffer), 0);

        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let tg = MTLSize::new(base.num_blocks as u64, 1, 1);
        if base.num_blocks > 0 {
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
        compute_encoder.set_buffer(1, Some(args.partials_buffer), 0);

        // Compute sizes
        let num_blocks = ((args.t + 255) / 256) as u32;
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
        compute_encoder.set_buffer(0, Some(args.partials_buffer), 0);
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
                // Not supported in v1; fallback to f32 logits path
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

// ---- Finalize Kernel ----
#[derive(Debug, thiserror::Error)]
pub enum MoeFinalizeError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
}

pub struct MoeFinalizeKernel {
    pipeline_f16: MTLComputePipelineState,
    pipeline_bf16: MTLComputePipelineState,
}

#[derive(Debug)]
pub struct MoeFinalizeArguments<'a> {
    pub tok2row_buffer: &'a MTLBuffer, // [T*K] i32
    pub probs_buffer: &'a MTLBuffer,   // [T*K] f16/bf16
    pub y_partial_buffer: &'a MTLBuffer, // [sum_k, d_model] f16
    pub y_out_buffer: &'a MTLBuffer,   // [T, d_model] f16
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
        Ok(Self {
            pipeline_f16,
            pipeline_bf16,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        args: MoeFinalizeArguments,
        dtype: KernelDataType,
    ) -> Result<(), MoeFinalizeError> {
        match dtype {
            KernelDataType::Float16 => {
                encoder.set_compute_pipeline_state(&self.pipeline_f16);
            },
            KernelDataType::BFloat16 => {
                encoder.set_compute_pipeline_state(&self.pipeline_bf16);
            },
            KernelDataType::Float32 => {
                // Not used for finalize in v1
                encoder.set_compute_pipeline_state(&self.pipeline_f16);
            },
        }
        encoder.set_buffer(0, Some(args.tok2row_buffer), 0);
        encoder.set_buffer(1, Some(args.probs_buffer), 0);
        encoder.set_buffer(2, Some(args.y_partial_buffer), 0);
        encoder.set_buffer(3, Some(args.y_out_buffer), 0);
        let t_u = args.t as u32;
        let dm_u = args.d_model as u32;
        let k_u = args.k as u32;
        encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            &t_u as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            &dm_u as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            size_of::<u32>() as u64,
            &k_u as *const u32 as *const _,
        );

        // Launch tiles over (N tiles, M tiles)
        const BM: usize = 32;
        const BN: usize = 64;
        let num_tiles_n = (args.d_model + BN - 1) / BN;
        let num_tiles_m = (args.t + BM - 1) / BM;
        let threads_per_threadgroup = MTLSize::new(128, 1, 1);
        if num_tiles_m > 0 && num_tiles_n > 0 {
            let threadgroups =
                MTLSize::new(num_tiles_n as u64, num_tiles_m as u64, 1);
            encoder
                .dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        }
        Ok(())
    }
}
