use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, CommandBufferRef,
    ComputePipelineState as MTLComputePipelineState, FunctionConstantValues,
    MTLDataType, MTLSize,
};

use super::{
    MoePassARowMapArguments, MoePassATileBuildArguments,
    MoePassATileCountsArguments, MoePassATileDispatchArguments,
    MoePassATileKernel, MoePassATileScanArguments, MoeTileCountsArguments,
    MoeTileDispatchArguments, MoeTileError, MoeTileMapBuildArguments,
    MoeTileMapKernel, MoeTileScanArguments, dtype_index, dtype_suffix,
};
use crate::backends::metal::{KernelDataType, MTLContext, MTLError};

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
    pub x_perm_buffer: &'a MTLBuffer, // [sum_k, d_model] - permuted input
    pub expert_offsets: &'a MTLBuffer, // [E+1] - expert segment offsets
    pub w13_all: &'a MTLBuffer, // [E, 2*d_ff, d_model] - transposed up projection weights
    pub w2_all: &'a MTLBuffer, // [E, d_model, d_ff] - transposed down projection weights
    pub y_partial: &'a MTLBuffer, // [sum_k, d_model] - output buffer
    pub up_biases: &'a MTLBuffer, // [E, 2*d_ff] - up projection biases
    pub down_biases: &'a MTLBuffer, // [E, d_model] - down projection biases
    pub tile_counts: &'a MTLBuffer, // [E]
    pub tile_row_offsets: &'a MTLBuffer, // [E+1]
    pub tile_map: &'a MTLBuffer, // [max_tiles * 3]
    pub total_tiles: &'a MTLBuffer, // [2]
    pub dispatch_args: &'a MTLBuffer, // [3]
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
    pass_a_indirect: Vec<Vec<MTLComputePipelineState>>, // [gate][dtype]
    fused_down: Vec<MTLComputePipelineState>,           // [dtype]
}

pub struct MoeExpertsTwoPassPrefillKernel {
    tile_map: MoeTileMapKernel,
    pass_a_indirect: Vec<Vec<MTLComputePipelineState>>, // [gate][dtype]
    pass_b_indirect: Vec<MTLComputePipelineState>,      // [dtype]
}

#[derive(Debug)]
pub struct MoeExpertsTwoPassArguments<'a> {
    pub x_perm_buffer: &'a MTLBuffer,
    pub expert_offsets: &'a MTLBuffer,
    pub row_expert_map: &'a MTLBuffer,
    pub hidden_buffer: &'a MTLBuffer,
    pub partial_buffer: &'a MTLBuffer,
    pub output_buffer: &'a MTLBuffer,
    pub w13_all: &'a MTLBuffer,
    pub w2_all: &'a MTLBuffer,
    pub up_biases: &'a MTLBuffer,
    pub down_biases: &'a MTLBuffer,
    pub tile_counts: &'a MTLBuffer,
    pub tile_offsets: &'a MTLBuffer,
    pub tile_map: &'a MTLBuffer,
    pub total_tiles: &'a MTLBuffer,
    pub dispatch_args: &'a MTLBuffer,
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
                let fcv = FunctionConstantValues::new();
                fcv.set_constant_value_at_index(
                    &gate as *const u32 as *const std::ffi::c_void,
                    MTLDataType::UInt,
                    30,
                );
                let tile_h: u32 = 512;
                fcv.set_constant_value_at_index(
                    &tile_h as *const u32 as *const std::ffi::c_void,
                    MTLDataType::UInt,
                    32,
                );
                let kernel_name = format!(
                    "moe_experts_decode_pass_a_indirect_{}",
                    dtype_suffix
                );
                pass_a_indirect[gate as usize].push(
                    ctx.compute_pipeline_state(&kernel_name, Some(&fcv))?,
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
        command_buffer: &CommandBufferRef,
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
        let encoder_a = command_buffer.new_compute_command_encoder();
        encoder_a.set_compute_pipeline_state(pass_a_pipeline);
        encoder_a.set_buffer(0, Some(args.x_perm_buffer), 0);
        encoder_a.set_buffer(1, Some(args.expert_offsets), 0);
        encoder_a.set_buffer(2, Some(args.w13_all), 0);
        encoder_a.set_buffer(3, Some(args.hidden_buffer), 0);
        encoder_a.set_buffer(4, Some(args.up_biases), 0);
        encoder_a.set_bytes(
            5,
            size_of::<u32>() as u64,
            &d_model_u32 as *const u32 as *const _,
        );
        encoder_a.set_bytes(
            6,
            size_of::<u32>() as u64,
            &d_ff_u32 as *const u32 as *const _,
        );
        encoder_a.set_bytes(
            7,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        encoder_a.set_bytes(
            8,
            size_of::<f32>() as u64,
            &args.gate_clip_min as *const f32 as *const _,
        );
        encoder_a.set_bytes(
            9,
            size_of::<f32>() as u64,
            &args.gate_clip_max as *const f32 as *const _,
        );
        encoder_a.set_bytes(
            10,
            size_of::<f32>() as u64,
            &args.up_clip_min as *const f32 as *const _,
        );
        encoder_a.set_bytes(
            11,
            size_of::<f32>() as u64,
            &args.up_clip_max as *const f32 as *const _,
        );
        encoder_a.set_bytes(
            12,
            size_of::<f32>() as u64,
            &args.silu_alpha as *const f32 as *const _,
        );
        encoder_a.set_buffer(13, Some(args.tile_map), 0);
        encoder_a.dispatch_thread_groups_indirect(
            args.dispatch_args,
            0,
            MTLSize::new(128, 1, 1),
        );
        encoder_a.end_encoding();
        let total_rows_u32 = args.total_rows as u32;
        let pass_b_pipeline = &self.fused_down[dtype_idx];
        let encoder_b = command_buffer.new_compute_command_encoder();
        encoder_b.set_compute_pipeline_state(pass_b_pipeline);
        encoder_b.set_buffer(0, Some(args.hidden_buffer), 0);
        encoder_b.set_buffer(1, Some(args.row_expert_map), 0);
        encoder_b.set_buffer(2, Some(args.w2_all), 0);
        encoder_b.set_buffer(3, Some(args.down_biases), 0);
        encoder_b.set_buffer(4, Some(args.output_buffer), 0);
        encoder_b.set_bytes(
            5,
            size_of::<u32>() as u64,
            &total_rows_u32 as *const u32 as *const _,
        );
        encoder_b.set_bytes(
            6,
            size_of::<u32>() as u64,
            &d_model_u32 as *const u32 as *const _,
        );
        encoder_b.set_bytes(
            7,
            size_of::<u32>() as u64,
            &d_ff_u32 as *const u32 as *const _,
        );
        encoder_b.set_bytes(
            8,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        const SIMDGROUPS_PER_TG: u32 = 8;
        const THREADS_PER_TG: u32 = 256;
        let col_blocks =
            (args.d_model as u32 + SIMDGROUPS_PER_TG - 1) / SIMDGROUPS_PER_TG;
        encoder_b.dispatch_thread_groups(
            MTLSize::new(col_blocks as u64, args.total_rows as u64, 1),
            MTLSize::new(THREADS_PER_TG as u64, 1, 1),
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
                let fcv = FunctionConstantValues::new();
                fcv.set_constant_value_at_index(
                    &gate as *const u32 as *const std::ffi::c_void,
                    MTLDataType::UInt,
                    30,
                );
                let kernel_name = format!(
                    "moe_two_pass_prefill_pass_a_indirect_{}",
                    dtype_suffix
                );
                pass_a_indirect[gate as usize].push(
                    ctx.compute_pipeline_state(&kernel_name, Some(&fcv))?,
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
        command_buffer: &CommandBufferRef,
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
        let hidden_bytes = (args.total_rows * args.d_ff * dtype_size) as u64;
        let blit_encoder = command_buffer.new_blit_command_encoder();
        blit_encoder.fill_buffer(
            args.hidden_buffer,
            metal::NSRange::new(0, hidden_bytes),
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
        const COL_TILE_FF: u32 = 64;
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
        let encoder_a = command_buffer.new_compute_command_encoder();
        encoder_a.set_compute_pipeline_state(pass_a_pipeline);
        encoder_a.set_buffer(0, Some(args.x_perm_buffer), 0);
        encoder_a.set_buffer(1, Some(args.expert_offsets), 0);
        encoder_a.set_buffer(2, Some(args.w13_all), 0);
        encoder_a.set_buffer(3, Some(args.up_biases), 0);
        encoder_a.set_buffer(4, Some(args.hidden_buffer), 0);
        encoder_a.set_bytes(
            5,
            size_of::<u32>() as u64,
            &d_model_u32 as *const u32 as *const _,
        );
        encoder_a.set_bytes(
            6,
            size_of::<u32>() as u64,
            &d_ff_u32 as *const u32 as *const _,
        );
        encoder_a.set_bytes(
            7,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        encoder_a.set_bytes(
            8,
            size_of::<f32>() as u64,
            &args.gate_clip_min as *const f32 as *const _,
        );
        encoder_a.set_bytes(
            9,
            size_of::<f32>() as u64,
            &args.gate_clip_max as *const f32 as *const _,
        );
        encoder_a.set_bytes(
            10,
            size_of::<f32>() as u64,
            &args.up_clip_min as *const f32 as *const _,
        );
        encoder_a.set_bytes(
            11,
            size_of::<f32>() as u64,
            &args.up_clip_max as *const f32 as *const _,
        );
        encoder_a.set_bytes(
            12,
            size_of::<f32>() as u64,
            &args.silu_alpha as *const f32 as *const _,
        );
        encoder_a.set_buffer(13, Some(args.tile_map), 0);
        // Match kernel config: WM=2, WN=2 => 4 SIMDgroups => 128 threads
        const SIMDGROUPS_PER_TG: u32 = 4;
        const THREADS_PER_TG: u32 = SIMDGROUPS_PER_TG * 32;
        encoder_a.dispatch_thread_groups_indirect(
            args.dispatch_args,
            0,
            MTLSize::new(THREADS_PER_TG as u64, 1, 1),
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
        let encoder_b = command_buffer.new_compute_command_encoder();
        encoder_b.set_compute_pipeline_state(pass_b_pipeline);
        encoder_b.set_buffer(0, Some(args.hidden_buffer), 0);
        encoder_b.set_buffer(1, Some(args.expert_offsets), 0);
        encoder_b.set_buffer(2, Some(args.w2_all), 0);
        encoder_b.set_buffer(3, Some(args.down_biases), 0);
        encoder_b.set_buffer(4, Some(args.output_buffer), 0);
        encoder_b.set_bytes(
            5,
            size_of::<u32>() as u64,
            &d_model_u32 as *const u32 as *const _,
        );
        encoder_b.set_bytes(
            6,
            size_of::<u32>() as u64,
            &d_ff_u32 as *const u32 as *const _,
        );
        encoder_b.set_bytes(
            7,
            size_of::<u32>() as u64,
            &e_u32 as *const u32 as *const _,
        );
        encoder_b.set_buffer(8, Some(args.tile_map), 0);
        encoder_b.dispatch_thread_groups_indirect(
            args.dispatch_args,
            0,
            MTLSize::new(THREADS_PER_TG as u64, 1, 1),
        );
        encoder_b.end_encoding();
        Ok(())
    }
}
