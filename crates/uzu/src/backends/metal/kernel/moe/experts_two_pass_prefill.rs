use metal::{MTLBlitCommandEncoderExt, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder};
use objc2::{
    __framework_prelude::{ProtocolObject, Retained},
    Message,
};

use crate::{
    DataType,
    backends::{
        common::kernel::{MoeExpertsPrefillPassAKernel, MoeExpertsPrefillPassBKernel},
        metal::{
            MetalContext, MetalError,
            kernel::{
                dsl::{MoeExpertsPrefillPassAMetalKernel, MoeExpertsPrefillPassBMetalKernel},
                moe::tiles_map::{
                    MoeTileCountsArguments, MoeTileDispatchArguments, MoeTileMapBuildArguments, MoeTileMapKernels,
                    MoeTileScanArguments,
                },
            },
        },
    },
};

const DTYPES: [DataType; 3] = [DataType::F16, DataType::BF16, DataType::F32];

pub struct MoeExpertsTwoPassPrefillBlock {
    tile_map: MoeTileMapKernels,
    pass_a_indirect: Vec<Vec<MoeExpertsPrefillPassAMetalKernel>>, // [gate][dtype]
    pass_b_indirect: Vec<MoeExpertsPrefillPassBMetalKernel>,      // [dtype]
}

impl MoeExpertsTwoPassPrefillBlock {
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let dtypes = [DataType::F16, DataType::BF16, DataType::F32];

        let mut pass_a_indirect = vec![];
        for gate in 0u32..4u32 {
            let mut kernels = vec![];
            for dtype in dtypes {
                let kernel = MoeExpertsPrefillPassAMetalKernel::new(ctx, dtype, gate)?;
                kernels.push(kernel);
            }
            pass_a_indirect.push(kernels);
        }

        let mut pass_b_indirect = vec![];
        for dtype in dtypes {
            let kernel = MoeExpertsPrefillPassBMetalKernel::new(ctx, dtype)?;
            pass_b_indirect.push(kernel);
        }

        Ok(Self {
            tile_map: MoeTileMapKernels::new(ctx)?,
            pass_a_indirect,
            pass_b_indirect,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        args: &MoeExpertsTwoPassArguments,
    ) {
        if args.total_rows == 0 {
            return;
        }

        let hidden_bytes = args.total_rows * args.d_ff * args.data_type.size_in_bytes();
        let blit_encoder = command_buffer.new_blit_command_encoder().expect("Failed to create blit command encoder");
        blit_encoder.fill_buffer_range_value(args.hidden_buffer, 0..hidden_bytes, 0);
        blit_encoder.end_encoding();

        self.tile_map.encode_counts(
            command_buffer,
            &MoeTileCountsArguments {
                offsets_buffer: args.expert_offsets,
                tile_counts_buffer: args.tile_counts,
                e: args.e,
            },
        );
        self.tile_map.encode_scan(
            command_buffer,
            &MoeTileScanArguments {
                tile_counts_buffer: args.tile_counts,
                tile_offsets_buffer: args.tile_offsets,
                total_tiles_buffer: args.total_tiles,
                e: args.e,
            },
        );
        self.tile_map.encode_build_map(
            command_buffer,
            &MoeTileMapBuildArguments {
                expert_offsets: args.expert_offsets,
                tile_offsets: args.tile_offsets,
                tile_counts: args.tile_counts,
                tile_map: args.tile_map,
                e: args.e,
            },
        );
        let d_model_u32 = args.d_model as u32;
        let d_ff_u32 = args.d_ff as u32;
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
        );

        let encoder_a =
            command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder A");
        let gate_idx = args.gating_code.min(3) as usize;
        let dtype_idx = DTYPES.iter().position(|t| *t == args.data_type).unwrap();
        let kernel_pass_a = &self.pass_a_indirect[gate_idx][dtype_idx];
        kernel_pass_a.encode(
            &args.x_perm_buffer.retain(),
            &args.expert_offsets.retain(),
            &args.w13_all.retain(),
            &args.up_biases.retain(),
            &args.hidden_buffer.retain(),
            args.d_model as u32,
            args.d_ff as u32,
            args.e as u32,
            args.gate_clip_min,
            args.gate_clip_max,
            args.up_clip_min,
            args.up_clip_max,
            args.silu_alpha,
            &args.tile_map.retain(),
            &args.dispatch_args.retain(),
            &encoder_a,
        );
        encoder_a.end_encoding();

        let dispatch_args = MoeTileDispatchArguments {
            total_tiles: args.total_tiles,
            dispatch_args: args.dispatch_args,
            num_tiles_x: n_tiles_model,
        };
        self.tile_map.encode_dispatch_args(command_buffer, &dispatch_args);

        let encoder_b =
            command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder B");
        let kernel_pass_b = &self.pass_b_indirect[dtype_idx];
        kernel_pass_b.encode(
            &args.hidden_buffer.retain(),
            &args.expert_offsets.retain(),
            &args.w2_all.retain(),
            &args.down_biases.retain(),
            &args.output_buffer.retain(),
            args.d_model as u32,
            args.d_ff as u32,
            args.e as u32,
            &args.tile_map.retain(),
            &args.dispatch_args.retain(),
            &encoder_b,
        );
        encoder_b.end_encoding();
    }
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
    pub data_type: DataType,
}
