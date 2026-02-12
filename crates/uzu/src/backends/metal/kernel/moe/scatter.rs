use crate::{
    DataType,
    backends::metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState, MTLContext,
        MTLSize, ProtocolObject, Retained, metal_extensions::ComputeEncoderSetValue,
    },
};

// ---- Scatter Buckets Kernels ----

#[derive(Debug, thiserror::Error)]
pub enum MoeScatterError {
    #[error("Metal error: {0}")]
    MetalError(#[from] crate::backends::metal::MTLError),
}

pub struct MoeScatterKernels {
    pipeline_bases: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_scatter_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_scatter_f32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_scatter_bf16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    // map variants
    pipeline_scatter_map_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_scatter_map_f32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_scatter_map_bf16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

#[derive(Debug)]
pub struct MoeBlockBasesArguments<'a> {
    pub partials_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [num_blocks * num_tiles * 512]
    pub block_bases_buffer: &'a ProtocolObject<dyn MTLBuffer>, // same shape as partials
    pub block_alloc_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [num_blocks * num_tiles * 512]
    pub e: usize,
    pub num_blocks: usize,
    pub num_tiles: usize,
}

#[derive(Debug)]
pub struct MoeScatterArguments<'a> {
    pub topk_ids_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub topk_probs_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub offsets_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub block_bases_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub block_alloc_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub out_ids_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub out_probs_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub t: usize,
    pub e: usize,
    pub k: usize,
    pub num_blocks: usize,
    pub num_tiles: usize,
}

#[derive(Debug)]
pub struct MoeScatterWithMapArguments<'a> {
    pub base: MoeScatterArguments<'a>,
    pub tok2row_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [T*K] int32, initialized to -1
}

impl MoeScatterKernels {
    pub fn new(mtl_context: &MTLContext) -> Result<Self, MoeScatterError> {
        let pipeline_bases = mtl_context.compute_pipeline_state("moe_block_bases_from_partials", None)?;
        let pipeline_scatter_f16 = mtl_context.compute_pipeline_state("moe_scatter_buckets_f16", None)?;
        let pipeline_scatter_f32 = mtl_context.compute_pipeline_state("moe_scatter_buckets_f32", None)?;
        let pipeline_scatter_bf16 = mtl_context.compute_pipeline_state("moe_scatter_buckets_bf16", None)?;
        let pipeline_scatter_map_f16 = mtl_context.compute_pipeline_state("moe_scatter_buckets_map_f16", None)?;
        let pipeline_scatter_map_f32 = mtl_context.compute_pipeline_state("moe_scatter_buckets_map_f32", None)?;
        let pipeline_scatter_map_bf16 = mtl_context.compute_pipeline_state("moe_scatter_buckets_map_bf16", None)?;

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
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        args: MoeBlockBasesArguments,
    ) -> Result<(), MoeScatterError> {
        let compute_encoder =
            command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
        compute_encoder.set_compute_pipeline_state(&self.pipeline_bases);
        compute_encoder.set_buffer(Some(args.partials_buffer), 0, 0);
        compute_encoder.set_buffer(Some(args.block_bases_buffer), 0, 1);
        compute_encoder.set_buffer(Some(args.block_alloc_buffer), 0, 2);

        let e_u32 = args.e as u32;
        let nb_u32 = args.num_blocks as u32;
        let nt_u32 = args.num_tiles as u32;
        let cap_u32: u32 = 0;
        compute_encoder.set_value(&e_u32, 3);
        compute_encoder.set_value(&nb_u32, 4);
        compute_encoder.set_value(&nt_u32, 5);
        compute_encoder.set_value(&cap_u32, 6);

        let total_entries = args.num_tiles * 512usize;
        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let tg = MTLSize::new((total_entries + 255) / 256, 1, 1);
        if total_entries > 0 {
            compute_encoder.dispatch_threadgroups(tg, threads_per_threadgroup);
        }
        compute_encoder.end_encoding();
        Ok(())
    }

    pub fn encode_scatter(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        args: MoeScatterArguments,
        dtype: DataType,
    ) -> Result<(), MoeScatterError> {
        let compute_encoder =
            command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
        // Select pipeline based on dtype
        match dtype {
            DataType::F16 => {
                compute_encoder.set_compute_pipeline_state(&self.pipeline_scatter_f16);
            },
            DataType::F32 => {
                compute_encoder.set_compute_pipeline_state(&self.pipeline_scatter_f32);
            },
            DataType::BF16 => {
                compute_encoder.set_compute_pipeline_state(&self.pipeline_scatter_bf16);
            },
            _ => panic!("Unsupported data type: {:?}", dtype),
        }
        compute_encoder.set_buffer(Some(args.topk_ids_buffer), 0, 0);
        compute_encoder.set_buffer(Some(args.topk_probs_buffer), 0, 1);
        compute_encoder.set_buffer(Some(args.offsets_buffer), 0, 2);
        compute_encoder.set_buffer(Some(args.block_bases_buffer), 0, 3);
        compute_encoder.set_buffer(Some(args.block_alloc_buffer), 0, 4);
        compute_encoder.set_buffer(Some(args.out_ids_buffer), 0, 5);
        compute_encoder.set_buffer(Some(args.out_probs_buffer), 0, 6);
        let t_u32 = args.t as u32;
        let e_u32 = args.e as u32;
        let k_u32 = args.k as u32;
        let nb_u32 = args.num_blocks as u32;
        let nt_u32 = args.num_tiles as u32;
        compute_encoder.set_value(&t_u32, 7);
        compute_encoder.set_value(&e_u32, 8);
        compute_encoder.set_value(&k_u32, 9);
        compute_encoder.set_value(&nb_u32, 10);
        compute_encoder.set_value(&nt_u32, 11);

        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let tg = MTLSize::new(args.num_blocks, 1, 1);
        if args.num_blocks > 0 {
            compute_encoder.dispatch_threadgroups(tg, threads_per_threadgroup);
        }
        compute_encoder.end_encoding();
        Ok(())
    }

    pub fn encode_scatter_with_map(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        args: MoeScatterWithMapArguments,
        dtype: DataType,
    ) -> Result<(), MoeScatterError> {
        let compute_encoder =
            command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
        let pipeline = match dtype {
            DataType::F16 => &self.pipeline_scatter_map_f16,
            DataType::F32 => &self.pipeline_scatter_map_f32,
            DataType::BF16 => &self.pipeline_scatter_map_bf16,
            _ => panic!("Unsupported data type: {:?}", dtype),
        };
        compute_encoder.set_compute_pipeline_state(pipeline);
        let base = &args.base;
        compute_encoder.set_buffer(Some(base.topk_ids_buffer), 0, 0);
        compute_encoder.set_buffer(Some(base.topk_probs_buffer), 0, 1);
        compute_encoder.set_buffer(Some(base.offsets_buffer), 0, 2);
        compute_encoder.set_buffer(Some(base.block_bases_buffer), 0, 3);
        compute_encoder.set_buffer(Some(base.block_alloc_buffer), 0, 4);
        compute_encoder.set_buffer(Some(base.out_ids_buffer), 0, 5);
        compute_encoder.set_buffer(Some(base.out_probs_buffer), 0, 6);
        let t_u32 = base.t as u32;
        let e_u32 = base.e as u32;
        let k_u32 = base.k as u32;
        let nb_u32 = base.num_blocks as u32;
        let nt_u32 = base.num_tiles as u32;
        compute_encoder.set_value(&t_u32, 7);
        compute_encoder.set_value(&e_u32, 8);
        compute_encoder.set_value(&k_u32, 9);
        compute_encoder.set_value(&nb_u32, 10);
        compute_encoder.set_value(&nt_u32, 11);
        compute_encoder.set_buffer(Some(args.tok2row_buffer), 0, 12);

        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let tg = MTLSize::new(base.num_blocks, 1, 1);
        if base.num_blocks > 0 {
            compute_encoder.dispatch_threadgroups(tg, threads_per_threadgroup);
        }
        compute_encoder.end_encoding();
        Ok(())
    }
}
