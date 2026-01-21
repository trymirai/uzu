use std::mem::size_of;

use crate::backends::metal::{
    BufferRef, CommandBufferRef, ComputeEncoderLegacy, ComputePipelineState,
    KernelDataType, MTLCommandBuffer, MTLCommandEncoder, MTLContext, mtl_size,
};

// ---- Scatter Buckets Kernels ----

#[derive(Debug, thiserror::Error)]
pub enum MoeScatterError {
    #[error("Metal error: {0}")]
    MetalError(#[from] crate::backends::metal::MTLError),
}

pub struct MoeScatterKernels {
    pipeline_bases: ComputePipelineState,
    pipeline_scatter_f16: ComputePipelineState,
    pipeline_scatter_f32: ComputePipelineState,
    pipeline_scatter_bf16: ComputePipelineState,
    // map variants
    pipeline_scatter_map_f16: ComputePipelineState,
    pipeline_scatter_map_f32: ComputePipelineState,
    pipeline_scatter_map_bf16: ComputePipelineState,
}

#[derive(Debug)]
pub struct MoeBlockBasesArguments<'a> {
    pub partials_buffer: BufferRef<'a>, // [num_blocks * num_tiles * 512]
    pub block_bases_buffer: BufferRef<'a>, // same shape as partials
    pub block_alloc_buffer: BufferRef<'a>, // [num_blocks * num_tiles * 512]
    pub e: usize,
    pub num_blocks: usize,
    pub num_tiles: usize,
}

#[derive(Debug)]
pub struct MoeScatterArguments<'a> {
    pub topk_ids_buffer: BufferRef<'a>,
    pub topk_probs_buffer: BufferRef<'a>,
    pub offsets_buffer: BufferRef<'a>,
    pub block_bases_buffer: BufferRef<'a>,
    pub block_alloc_buffer: BufferRef<'a>,
    pub out_ids_buffer: BufferRef<'a>,
    pub out_probs_buffer: BufferRef<'a>,
    pub t: usize,
    pub e: usize,
    pub k: usize,
    pub num_blocks: usize,
    pub num_tiles: usize,
}

#[derive(Debug)]
pub struct MoeScatterWithMapArguments<'a> {
    pub base: MoeScatterArguments<'a>,
    pub tok2row_buffer: BufferRef<'a>, // [T*K] int32, initialized to -1
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
        command_buffer: &CommandBufferRef,
        args: MoeBlockBasesArguments,
    ) -> Result<(), MoeScatterError> {
        let compute_encoder = command_buffer.new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
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
        let threads_per_threadgroup = mtl_size(256, 1, 1);
        let tg = mtl_size(((total_entries + 255) / 256) as u64, 1, 1);
        if total_entries > 0 {
            compute_encoder.dispatch_thread_groups(tg, threads_per_threadgroup);
        }
        compute_encoder.end_encoding();
        Ok(())
    }

    pub fn encode_scatter(
        &self,
        command_buffer: &CommandBufferRef,
        args: MoeScatterArguments,
        dtype: KernelDataType,
    ) -> Result<(), MoeScatterError> {
        let compute_encoder = command_buffer.new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
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

        let threads_per_threadgroup = mtl_size(256, 1, 1);
        let tg = mtl_size(args.num_blocks as u64, 1, 1);
        if args.num_blocks > 0 {
            compute_encoder.dispatch_thread_groups(tg, threads_per_threadgroup);
        }
        compute_encoder.end_encoding();
        Ok(())
    }

    pub fn encode_scatter_with_map(
        &self,
        command_buffer: &CommandBufferRef,
        args: MoeScatterWithMapArguments,
        dtype: KernelDataType,
    ) -> Result<(), MoeScatterError> {
        let compute_encoder = command_buffer.new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
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

        let threads_per_threadgroup = mtl_size(256, 1, 1);
        let tg = mtl_size(base.num_blocks as u64, 1, 1);
        if base.num_blocks > 0 {
            compute_encoder.dispatch_thread_groups(tg, threads_per_threadgroup);
        }
        compute_encoder.end_encoding();
        Ok(())
    }
}
