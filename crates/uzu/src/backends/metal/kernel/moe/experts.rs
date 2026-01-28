use std::{mem::size_of, ptr::NonNull};

use super::{
    MoePassARowMapArguments, MoePassATileBuildArguments,
    MoePassATileCountsArguments, MoePassATileDispatchArguments,
    MoePassATileKernel, MoePassATileScanArguments, MoeTileCountsArguments,
    MoeTileDispatchArguments, MoeTileError, MoeTileMapBuildArguments,
    MoeTileMapKernel, MoeTileScanArguments, dtype_index, dtype_suffix,
};
use crate::backends::metal::{
    KernelDataType, MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer,
    MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLContext, MTLDataType, MTLError, MTLFunctionConstantValues, MTLSize,
    NSRange, ProtocolObject, Retained,
};

#[derive(Debug, thiserror::Error)]
pub enum MoeScatterError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
}
#[derive(Debug, thiserror::Error)]
pub enum MoeExpertsError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("{0}")]
    Generic(String),
}

impl From<MoeTileError> for MoeExpertsError {
    fn from(err: MoeTileError) -> Self {
        match err {
            MoeTileError::MetalError(inner) => {
                MoeExpertsError::MetalError(inner)
            },
        }
    }
}

#[derive(Debug)]
pub struct MoeExpertsArguments<'a> {
    pub x_perm_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [sum_k, d_model] - permuted input
    pub expert_offsets: &'a ProtocolObject<dyn MTLBuffer>, // [E+1] - expert segment offsets
    pub w13_all: &'a ProtocolObject<dyn MTLBuffer>, // [E, 2*d_ff, d_model] - transposed up projection weights
    pub w2_all: &'a ProtocolObject<dyn MTLBuffer>, // [E, d_model, d_ff] - transposed down projection weights
    pub y_partial: &'a ProtocolObject<dyn MTLBuffer>, // [sum_k, d_model] - output buffer
    pub up_biases: &'a ProtocolObject<dyn MTLBuffer>, // [E, 2*d_ff] - up projection biases
    pub down_biases: &'a ProtocolObject<dyn MTLBuffer>, // [E, d_model] - down projection biases
    pub tile_counts: &'a ProtocolObject<dyn MTLBuffer>, // [E]
    pub tile_row_offsets: &'a ProtocolObject<dyn MTLBuffer>, // [E+1]
    pub tile_map: &'a ProtocolObject<dyn MTLBuffer>,    // [max_tiles * 3]
    pub total_tiles: &'a ProtocolObject<dyn MTLBuffer>, // [2]
    pub dispatch_args: &'a ProtocolObject<dyn MTLBuffer>, // [3]
    pub num_tiles_n: usize,
    pub t: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub e: usize,
    pub k: usize,
    pub gating_code: u32,
    pub gate_clip_min: f32,
    pub gate_clip_max: f32,
    pub up_clip_min: f32,
    pub up_clip_max: f32,
    pub silu_alpha: f32,
    pub data_type: KernelDataType,
}

pub struct MoeExpertsTwoPassDecodeKernel {
    pass_a_tile: MoePassATileKernel,
    pass_a_indirect:
        Vec<Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>>, // [gate][dtype]
    fused_down: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>, // [dtype]
}

pub struct MoeExpertsTwoPassPrefillKernel {
    tile_map: MoeTileMapKernel,
    pass_a_indirect:
        Vec<Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>>, // [gate][dtype]
    pass_b_indirect: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>, // [dtype]
}

#[derive(Debug)]
pub struct MoeExpertsTwoPassArguments<'a> {
    pub x_perm_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub expert_offsets: &'a ProtocolObject<dyn MTLBuffer>,
    pub row_expert_map: &'a ProtocolObject<dyn MTLBuffer>,
    pub hidden_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub output_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub w13_all: &'a ProtocolObject<dyn MTLBuffer>,
    pub w2_all: &'a ProtocolObject<dyn MTLBuffer>,
    pub up_biases: &'a ProtocolObject<dyn MTLBuffer>,
    pub down_biases: &'a ProtocolObject<dyn MTLBuffer>,
    pub tile_counts: &'a ProtocolObject<dyn MTLBuffer>,
    pub tile_offsets: &'a ProtocolObject<dyn MTLBuffer>,
    pub tile_map: &'a ProtocolObject<dyn MTLBuffer>,
    pub total_tiles: &'a ProtocolObject<dyn MTLBuffer>,
    pub dispatch_args: &'a ProtocolObject<dyn MTLBuffer>,
    pub total_rows: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub e: usize,
    pub num_tiles_k: u32,
    pub gating_code: u32,
    pub gate_clip_min: f32,
    pub gate_clip_max: f32,
    pub up_clip_min: f32,
    pub up_clip_max: f32,
    pub silu_alpha: f32,
    pub data_type: KernelDataType,
}

impl MoeExpertsTwoPassDecodeKernel {
    pub fn new(ctx: &MTLContext) -> Result<Self, MoeExpertsError> {
        let dtypes = [
            KernelDataType::Float16,
            KernelDataType::BFloat16,
            KernelDataType::Float32,
        ];
        let mut pass_a_indirect = vec![Vec::with_capacity(dtypes.len()); 4];
        for gate in 0u32..4u32 {
            for dtype in &dtypes {
                let dtype_suffix = dtype_suffix(*dtype);
                let fcv = MTLFunctionConstantValues::new();
                fcv.set_constant_value_type_at_index(
                    NonNull::from(&gate).cast(),
                    MTLDataType::UInt,
                    30,
                );
                let tile_h: u32 = 512;
                fcv.set_constant_value_type_at_index(
                    NonNull::from(&tile_h).cast(),
                    MTLDataType::UInt,
                    32,
                );
                let kernel_name = format!(
                    "moe_experts_decode_pass_a_indirect_{}",
                    dtype_suffix
                );
                let cache_key =
                    format!("{}_gate_{}_tile_{}", kernel_name, gate, tile_h);
                pass_a_indirect[gate as usize].push(
                    ctx.compute_pipeline_state_cached(
                        &cache_key,
                        &kernel_name,
                        Some(&fcv),
                    )?,
                );
            }
        }
        let mut fused_down = Vec::with_capacity(dtypes.len());
        for dtype in &dtypes {
            let dtype_suffix = dtype_suffix(*dtype);
            let kernel_name =
                format!("moe_experts_decode_down_fused_2d_{}", dtype_suffix);
            fused_down.push(ctx.compute_pipeline_state(&kernel_name, None)?);
        }
        Ok(Self {
            pass_a_tile: MoePassATileKernel::new(ctx)?,
            pass_a_indirect,
            fused_down,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        args: MoeExpertsTwoPassArguments,
    ) -> Result<(), MoeExpertsError> {
        if args.total_rows == 0 {
            return Ok(());
        }
        let gate_idx = (args.gating_code.min(3)) as usize;
        let dtype_idx = dtype_index(args.data_type);
        const BLOCK_M: u32 = 4;
        let h_blocks = (args.d_ff as u32 + BLOCK_M - 1) / BLOCK_M;
        self.pass_a_tile.encode_counts(
            command_buffer,
            &MoePassATileCountsArguments {
                expert_offsets: args.expert_offsets,
                tile_counts: args.tile_counts,
                e: args.e,
                h_blocks,
            },
        )?;
        self.pass_a_tile.encode_scan(
            command_buffer,
            &MoePassATileScanArguments {
                tile_counts: args.tile_counts,
                tile_offsets: args.tile_offsets,
                total_tiles: args.total_tiles,
                e: args.e,
            },
        )?;
        self.pass_a_tile.encode_row_map(
            command_buffer,
            &MoePassARowMapArguments {
                expert_offsets: args.expert_offsets,
                row_expert_map: args.row_expert_map,
                total_rows: args.total_rows,
                e: args.e,
            },
        )?;
        self.pass_a_tile.encode_build_map(
            command_buffer,
            &MoePassATileBuildArguments {
                expert_offsets: args.expert_offsets,
                tile_offsets: args.tile_offsets,
                row_expert_map: args.row_expert_map,
                tile_map: args.tile_map,
                total_rows: args.total_rows,
                h_blocks,
            },
        )?;
        self.pass_a_tile.encode_dispatch_args(
            command_buffer,
            &MoePassATileDispatchArguments {
                total_tiles: args.total_tiles,
                dispatch_args: args.dispatch_args,
                num_tiles_y: 1,
            },
        )?;
        let d_model_u32 = args.d_model as u32;
        let d_ff_u32 = args.d_ff as u32;
        let e_u32 = args.e as u32;
        let pass_a_pipeline = &self.pass_a_indirect[gate_idx][dtype_idx];
        let encoder_a = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        encoder_a.set_compute_pipeline_state(pass_a_pipeline);
        encoder_a.set_buffer(Some(args.x_perm_buffer), 0, 0);
        encoder_a.set_buffer(Some(args.expert_offsets), 0, 1);
        encoder_a.set_buffer(Some(args.w13_all), 0, 2);
        encoder_a.set_buffer(Some(args.hidden_buffer), 0, 3);
        encoder_a.set_buffer(Some(args.up_biases), 0, 4);
        unsafe {
            encoder_a.set_bytes(
                NonNull::new_unchecked(&d_model_u32 as *const _ as *mut _),
                size_of::<u32>(),
                5,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(&d_ff_u32 as *const _ as *mut _),
                size_of::<u32>(),
                6,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(&e_u32 as *const _ as *mut _),
                size_of::<u32>(),
                7,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(
                    &args.gate_clip_min as *const _ as *mut _,
                ),
                size_of::<f32>(),
                8,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(
                    &args.gate_clip_max as *const _ as *mut _,
                ),
                size_of::<f32>(),
                9,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(&args.up_clip_min as *const _ as *mut _),
                size_of::<f32>(),
                10,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(&args.up_clip_max as *const _ as *mut _),
                size_of::<f32>(),
                11,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(&args.silu_alpha as *const _ as *mut _),
                size_of::<f32>(),
                12,
            );
        }
        encoder_a.set_buffer(Some(args.tile_map), 0, 13);
        encoder_a.dispatch_threadgroups_indirect(
            args.dispatch_args,
            0,
            MTLSize::new(128, 1, 1),
        );
        encoder_a.end_encoding();
        let total_rows_u32 = args.total_rows as u32;
        let pass_b_pipeline = &self.fused_down[dtype_idx];
        let encoder_b = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        encoder_b.set_compute_pipeline_state(pass_b_pipeline);
        encoder_b.set_buffer(Some(args.hidden_buffer), 0, 0);
        encoder_b.set_buffer(Some(args.row_expert_map), 0, 1);
        encoder_b.set_buffer(Some(args.w2_all), 0, 2);
        encoder_b.set_buffer(Some(args.down_biases), 0, 3);
        encoder_b.set_buffer(Some(args.output_buffer), 0, 4);
        unsafe {
            encoder_b.set_bytes(
                NonNull::new_unchecked(&total_rows_u32 as *const _ as *mut _),
                size_of::<u32>(),
                5,
            );
            encoder_b.set_bytes(
                NonNull::new_unchecked(&d_model_u32 as *const _ as *mut _),
                size_of::<u32>(),
                6,
            );
            encoder_b.set_bytes(
                NonNull::new_unchecked(&d_ff_u32 as *const _ as *mut _),
                size_of::<u32>(),
                7,
            );
            encoder_b.set_bytes(
                NonNull::new_unchecked(&e_u32 as *const _ as *mut _),
                size_of::<u32>(),
                8,
            );
        }
        const SIMDGROUPS_PER_TG: u32 = 8;
        const THREADS_PER_TG: u32 = 256;
        let col_blocks =
            (args.d_model as u32 + SIMDGROUPS_PER_TG - 1) / SIMDGROUPS_PER_TG;
        encoder_b.dispatch_threadgroups(
            MTLSize::new(col_blocks as usize, args.total_rows, 1),
            MTLSize::new(THREADS_PER_TG as usize, 1, 1),
        );
        encoder_b.end_encoding();
        Ok(())
    }
}

impl MoeExpertsTwoPassPrefillKernel {
    pub fn new(ctx: &MTLContext) -> Result<Self, MoeExpertsError> {
        let dtypes = [
            KernelDataType::Float16,
            KernelDataType::BFloat16,
            KernelDataType::Float32,
        ];
        let mut pass_a_indirect = vec![Vec::with_capacity(dtypes.len()); 4];
        for gate in 0u32..4u32 {
            for dtype in &dtypes {
                let dtype_suffix = dtype_suffix(*dtype);
                let fcv = MTLFunctionConstantValues::new();
                fcv.set_constant_value_type_at_index(
                    NonNull::from(&gate).cast(),
                    MTLDataType::UInt,
                    30,
                );
                let kernel_name = format!(
                    "moe_two_pass_prefill_pass_a_indirect_{}",
                    dtype_suffix
                );
                let cache_key = format!("{}_gate_{}", kernel_name, gate);
                pass_a_indirect[gate as usize].push(
                    ctx.compute_pipeline_state_cached(
                        &cache_key,
                        &kernel_name,
                        Some(&fcv),
                    )?,
                );
            }
        }
        let mut pass_b_indirect = Vec::with_capacity(dtypes.len());
        for dtype in &dtypes {
            let dtype_suffix = dtype_suffix(*dtype);
            let kernel_name = format!(
                "moe_two_pass_prefill_pass_b_indirect_{}",
                dtype_suffix
            );
            pass_b_indirect
                .push(ctx.compute_pipeline_state(&kernel_name, None)?);
        }
        Ok(Self {
            tile_map: MoeTileMapKernel::new(ctx)?,
            pass_a_indirect,
            pass_b_indirect,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        args: MoeExpertsTwoPassArguments,
    ) -> Result<(), MoeExpertsError> {
        if args.total_rows == 0 {
            return Ok(());
        }
        let gate_idx = (args.gating_code.min(3)) as usize;
        let dtype_idx = dtype_index(args.data_type);
        let dtype_size = match args.data_type {
            KernelDataType::BFloat16 | KernelDataType::Float16 => 2,
            KernelDataType::Float32 => 4,
        };
        let hidden_bytes = args.total_rows * args.d_ff * dtype_size;
        let blit_encoder = command_buffer
            .new_blit_command_encoder()
            .expect("Failed to create blit command encoder");
        blit_encoder.fill_buffer_range_value(
            args.hidden_buffer,
            NSRange::new(0, hidden_bytes),
            0,
        );
        blit_encoder.end_encoding();
        self.tile_map.encode_counts(
            command_buffer,
            &MoeTileCountsArguments {
                offsets_buffer: args.expert_offsets,
                tile_counts_buffer: args.tile_counts,
                e: args.e,
            },
        )?;
        self.tile_map.encode_scan(
            command_buffer,
            &MoeTileScanArguments {
                tile_counts_buffer: args.tile_counts,
                tile_offsets_buffer: args.tile_offsets,
                total_tiles_buffer: args.total_tiles,
                e: args.e,
            },
        )?;
        self.tile_map.encode_build_map(
            command_buffer,
            &MoeTileMapBuildArguments {
                expert_offsets: args.expert_offsets,
                tile_offsets: args.tile_offsets,
                tile_counts: args.tile_counts,
                tile_map: args.tile_map,
                e: args.e,
            },
        )?;
        let d_model_u32 = args.d_model as u32;
        let d_ff_u32 = args.d_ff as u32;
        let e_u32 = args.e as u32;
        const COL_TILE_FF: u32 = 32; // Must match PASSA_BN in Metal kernel
        const COL_TILE_MODEL: u32 = 64;
        let n_tiles_ff = if d_ff_u32 == 0 {
            0
        } else {
            (d_ff_u32 + COL_TILE_FF - 1) / COL_TILE_FF
        };
        let n_tiles_model = if d_model_u32 == 0 {
            0
        } else {
            (d_model_u32 + COL_TILE_MODEL - 1) / COL_TILE_MODEL
        };
        self.tile_map.encode_dispatch_args(
            command_buffer,
            &MoeTileDispatchArguments {
                total_tiles: args.total_tiles,
                dispatch_args: args.dispatch_args,
                num_tiles_x: n_tiles_ff,
            },
        )?;
        let pass_a_pipeline = &self.pass_a_indirect[gate_idx][dtype_idx];
        let encoder_a = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        encoder_a.set_compute_pipeline_state(pass_a_pipeline);
        encoder_a.set_buffer(Some(args.x_perm_buffer), 0, 0);
        encoder_a.set_buffer(Some(args.expert_offsets), 0, 1);
        encoder_a.set_buffer(Some(args.w13_all), 0, 2);
        encoder_a.set_buffer(Some(args.up_biases), 0, 3);
        encoder_a.set_buffer(Some(args.hidden_buffer), 0, 4);
        unsafe {
            encoder_a.set_bytes(
                NonNull::new_unchecked(&d_model_u32 as *const _ as *mut _),
                size_of::<u32>(),
                5,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(&d_ff_u32 as *const _ as *mut _),
                size_of::<u32>(),
                6,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(&e_u32 as *const _ as *mut _),
                size_of::<u32>(),
                7,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(
                    &args.gate_clip_min as *const _ as *mut _,
                ),
                size_of::<f32>(),
                8,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(
                    &args.gate_clip_max as *const _ as *mut _,
                ),
                size_of::<f32>(),
                9,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(&args.up_clip_min as *const _ as *mut _),
                size_of::<f32>(),
                10,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(&args.up_clip_max as *const _ as *mut _),
                size_of::<f32>(),
                11,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(&args.silu_alpha as *const _ as *mut _),
                size_of::<f32>(),
                12,
            );
        }
        encoder_a.set_buffer(Some(args.tile_map), 0, 13);
        // Match kernel config: WM=2, WN=2 => 4 SIMDgroups => 128 threads
        const SIMDGROUPS_PER_TG: u32 = 4;
        const THREADS_PER_TG: u32 = SIMDGROUPS_PER_TG * 32;
        encoder_a.dispatch_threadgroups_indirect(
            args.dispatch_args,
            0,
            MTLSize::new(THREADS_PER_TG as usize, 1, 1),
        );
        encoder_a.end_encoding();
        self.tile_map.encode_dispatch_args(
            command_buffer,
            &MoeTileDispatchArguments {
                total_tiles: args.total_tiles,
                dispatch_args: args.dispatch_args,
                num_tiles_x: n_tiles_model,
            },
        )?;
        let pass_b_pipeline = &self.pass_b_indirect[dtype_idx];
        let encoder_b = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        encoder_b.set_compute_pipeline_state(pass_b_pipeline);
        encoder_b.set_buffer(Some(args.hidden_buffer), 0, 0);
        encoder_b.set_buffer(Some(args.expert_offsets), 0, 1);
        encoder_b.set_buffer(Some(args.w2_all), 0, 2);
        encoder_b.set_buffer(Some(args.down_biases), 0, 3);
        encoder_b.set_buffer(Some(args.output_buffer), 0, 4);
        unsafe {
            encoder_b.set_bytes(
                NonNull::new_unchecked(&d_model_u32 as *const _ as *mut _),
                size_of::<u32>(),
                5,
            );
            encoder_b.set_bytes(
                NonNull::new_unchecked(&d_ff_u32 as *const _ as *mut _),
                size_of::<u32>(),
                6,
            );
            encoder_b.set_bytes(
                NonNull::new_unchecked(&e_u32 as *const _ as *mut _),
                size_of::<u32>(),
                7,
            );
        }
        encoder_b.set_buffer(Some(args.tile_map), 0, 8);
        encoder_b.dispatch_threadgroups_indirect(
            args.dispatch_args,
            0,
            MTLSize::new(THREADS_PER_TG as usize, 1, 1),
        );
        encoder_b.end_encoding();
        Ok(())
    }
}

/// Arguments for single-token MoE decode (T=1 optimized path)
#[derive(Debug)]
pub struct MoeExpertsSingleDecodeArguments<'a> {
    /// Input activation [d_model]
    pub x: &'a ProtocolObject<dyn MTLBuffer>,
    /// Top-K expert indices from router [K]
    pub topk_ids: &'a ProtocolObject<dyn MTLBuffer>,
    /// Top-K probabilities from router [K]
    pub topk_probs: &'a ProtocolObject<dyn MTLBuffer>,
    /// Up/gate projection weights [E, 2*d_ff, d_model]
    pub w13_all: &'a ProtocolObject<dyn MTLBuffer>,
    /// Down projection weights [E, d_model, d_ff]
    pub w2_all: &'a ProtocolObject<dyn MTLBuffer>,
    /// Up/gate biases [E, 2*d_ff]
    pub up_biases: &'a ProtocolObject<dyn MTLBuffer>,
    /// Down biases [E, d_model]
    pub down_biases: &'a ProtocolObject<dyn MTLBuffer>,
    /// Hidden buffer [K, d_ff] - intermediate storage (f32)
    pub hidden: &'a ProtocolObject<dyn MTLBuffer>,
    /// Final output [d_model]
    pub y: &'a ProtocolObject<dyn MTLBuffer>,
    /// Model dimension
    pub d_model: usize,
    /// FFN hidden dimension
    pub d_ff: usize,
    /// Number of active experts per token
    pub k: usize,
    /// Gating activation: 0=GELU, 1=SiLU, 2=SwiGLU, 3=GEGLU
    pub gating_code: u32,
    /// SiLU activation alpha parameter
    pub silu_alpha: f32,
    /// Gate clipping min
    pub gate_clip_min: f32,
    /// Gate clipping max
    pub gate_clip_max: f32,
    /// Up clipping min
    pub up_clip_min: f32,
    /// Up clipping max
    pub up_clip_max: f32,
    /// Data type
    pub data_type: KernelDataType,
}

pub struct MoeExpertsSingleDecodeKernel {
    pass_a: Vec<Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>>, // [gate][dtype]
    pass_b: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>, // [dtype]
}

impl MoeExpertsSingleDecodeKernel {
    pub fn new(ctx: &MTLContext) -> Result<Self, MoeExpertsError> {
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
                let fcv = MTLFunctionConstantValues::new();
                fcv.set_constant_value_type_at_index(
                    NonNull::from(&gate).cast(),
                    MTLDataType::UInt,
                    30,
                );
                let kernel_name =
                    format!("moe_experts_decode_single_pass_a_{}", suffix);
                let cache_key = format!("{}_gate_{}", kernel_name, gate);
                pass_a[gate as usize].push(ctx.compute_pipeline_state_cached(
                    &cache_key,
                    &kernel_name,
                    Some(&fcv),
                )?);
            }
        }

        // Pass B (fused with finalize): no function constants needed
        let mut pass_b = Vec::with_capacity(dtypes.len());
        for dtype in &dtypes {
            let suffix = dtype_suffix(*dtype);
            let kernel_name =
                format!("moe_experts_decode_single_pass_b_{}", suffix);
            pass_b.push(ctx.compute_pipeline_state(&kernel_name, None)?);
        }

        Ok(Self {
            pass_a,
            pass_b,
        })
    }

    /// Encode the single-token decode pipeline
    ///
    /// Dispatches:
    /// 1. Pass A: grid = (h_blocks, K) - K parallel expert up+gate projections
    /// 2. Pass B: grid = (d_blocks, 1) - computes final y directly (fused finalize)
    pub fn encode(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        args: MoeExpertsSingleDecodeArguments,
    ) -> Result<(), MoeExpertsError> {
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
            let encoder = command_buffer
                .new_compute_command_encoder()
                .expect("Failed to create compute command encoder");
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(Some(args.x), 0, 0);
            encoder.set_buffer(Some(args.topk_ids), 0, 1);
            encoder.set_buffer(Some(args.w13_all), 0, 2);
            encoder.set_buffer(Some(args.up_biases), 0, 3);
            encoder.set_buffer(Some(args.hidden), 0, 4);
            unsafe {
                encoder.set_bytes(
                    NonNull::new_unchecked(&d_model_u32 as *const _ as *mut _),
                    size_of::<u32>(),
                    5,
                );
                encoder.set_bytes(
                    NonNull::new_unchecked(&d_ff_u32 as *const _ as *mut _),
                    size_of::<u32>(),
                    6,
                );
                encoder.set_bytes(
                    NonNull::new_unchecked(&k_u32 as *const _ as *mut _),
                    size_of::<u32>(),
                    7,
                );
                encoder.set_bytes(
                    NonNull::new_unchecked(
                        &args.silu_alpha as *const _ as *mut _,
                    ),
                    size_of::<f32>(),
                    8,
                );
                encoder.set_bytes(
                    NonNull::new_unchecked(
                        &args.gate_clip_min as *const _ as *mut _,
                    ),
                    size_of::<f32>(),
                    9,
                );
                encoder.set_bytes(
                    NonNull::new_unchecked(
                        &args.gate_clip_max as *const _ as *mut _,
                    ),
                    size_of::<f32>(),
                    10,
                );
                encoder.set_bytes(
                    NonNull::new_unchecked(
                        &args.up_clip_min as *const _ as *mut _,
                    ),
                    size_of::<f32>(),
                    11,
                );
                encoder.set_bytes(
                    NonNull::new_unchecked(
                        &args.up_clip_max as *const _ as *mut _,
                    ),
                    size_of::<f32>(),
                    12,
                );
            }
            encoder.dispatch_threadgroups(
                MTLSize::new(h_blocks as usize, args.k, 1),
                MTLSize::new(PASSA_THREADS as usize, 1, 1),
            );
            encoder.end_encoding();
        }

        // Pass B: 8 simdgroups (256 threads), outputs final y directly
        const PASSB_BLOCK_D: u32 = 8;
        const PASSB_THREADS: u32 = 256;
        let d_blocks = (d_model_u32 + PASSB_BLOCK_D - 1) / PASSB_BLOCK_D;

        // Pass B: hidden @ W2[expert] -> y (with weighted sum)
        {
            let pipeline = &self.pass_b[dtype_idx];
            let encoder = command_buffer
                .new_compute_command_encoder()
                .expect("Failed to create compute command encoder");
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(Some(args.hidden), 0, 0);
            encoder.set_buffer(Some(args.topk_ids), 0, 1);
            encoder.set_buffer(Some(args.topk_probs), 0, 2);
            encoder.set_buffer(Some(args.w2_all), 0, 3);
            encoder.set_buffer(Some(args.down_biases), 0, 4);
            encoder.set_buffer(Some(args.y), 0, 5);
            unsafe {
                encoder.set_bytes(
                    NonNull::new_unchecked(&d_model_u32 as *const _ as *mut _),
                    size_of::<u32>(),
                    6,
                );
                encoder.set_bytes(
                    NonNull::new_unchecked(&d_ff_u32 as *const _ as *mut _),
                    size_of::<u32>(),
                    7,
                );
                encoder.set_bytes(
                    NonNull::new_unchecked(&k_u32 as *const _ as *mut _),
                    size_of::<u32>(),
                    8,
                );
            }
            encoder.dispatch_threadgroups(
                MTLSize::new(d_blocks as usize, 1, 1),
                MTLSize::new(PASSB_THREADS as usize, 1, 1),
            );
            encoder.end_encoding();
        }

        Ok(())
    }
}
