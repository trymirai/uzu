use metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
};
use objc2::{
    __framework_prelude::{ProtocolObject, Retained},
    Message,
};
use objc2_foundation::NSRange;

use crate::{
    DataType,
    backends::{
        common::kernel::{
            MoeTwoPassPrefillPassAIndirectKernel,
            MoeTwoPassPrefillPassBIndirectKernel,
        },
        metal::{
            KernelDataType, MTLContext, MTLError,
            kernel::{
                MoeExpertsTwoPassArguments,
                dsl::{
                    MoeExpertsDecodeDownFused2DMetalKernel,
                    MoeExpertsDecodePassAMetalKernel,
                    MoeTwoPassPrefillPassAIndirectMetalKernel,
                    MoeTwoPassPrefillPassBIndirectMetalKernel,
                },
                moe::tiles_map::{
                    MoeTileCountsArguments, MoeTileDispatchArguments,
                    MoeTileMapBuildArguments, MoeTileMapKernels,
                    MoeTileScanArguments,
                },
            },
        },
    },
};

static DTYPES: [KernelDataType; 3] = [
    KernelDataType::Float16,
    KernelDataType::BFloat16,
    KernelDataType::Float32,
];

pub struct MoeExpertsTwoPassPrefillKernels {
    tile_map: MoeTileMapKernels,
    pass_a: Vec<Vec<MoeTwoPassPrefillPassAIndirectMetalKernel>>,
    pass_b: Vec<MoeTwoPassPrefillPassBIndirectMetalKernel>,
}

impl MoeExpertsTwoPassPrefillKernels {
    pub fn new(ctx: &MTLContext) -> Result<Self, MTLError> {
        let mut pass_a = vec![];
        for gate in 0u32..4u32 {
            let mut kernels = vec![];
            for dtype in &DTYPES {
                let data_type: DataType = (*dtype).into();
                let kernel = MoeTwoPassPrefillPassAIndirectMetalKernel::new(
                    ctx, data_type, gate,
                )?;
                kernels.push(kernel);
            }
            pass_a.push(kernels);
        }

        let mut pass_b = Vec::with_capacity(DTYPES.len());
        for dtype in &DTYPES {
            let data_type: DataType = (*dtype).into();
            let kernel =
                MoeTwoPassPrefillPassBIndirectMetalKernel::new(ctx, data_type)?;
            pass_b.push(kernel);
        }

        Ok(Self {
            tile_map: MoeTileMapKernels::new(ctx)?,
            pass_a,
            pass_b,
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

        let gate_idx = args.gating_code.min(3) as usize;
        let dtype_idx = DTYPES
            .iter()
            .position(|&t| t == args.data_type)
            .expect("Invalid dtype index");
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

        let pass_a_kernel = &self.pass_a[gate_idx][dtype_idx];
        let encoder_a = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        pass_a_kernel.encode(
            &args.x_perm_buffer.retain(),
            &args.expert_offsets.retain(),
            &args.w13_all.retain(),
            &args.up_biases.retain(),
            &args.tile_map.retain(),
            &args.hidden_buffer.retain(),
            args.d_model as u32,
            args.d_ff as u32,
            args.e as u32,
            args.gate_clip_min,
            args.gate_clip_max,
            args.up_clip_min,
            args.up_clip_max,
            args.silu_alpha,
            &args.dispatch_args.retain(),
            &encoder_a,
        );
        encoder_a.end_encoding();

        self.tile_map.encode_dispatch_args(
            command_buffer,
            &MoeTileDispatchArguments {
                total_tiles: args.total_tiles,
                dispatch_args: args.dispatch_args,
                num_tiles_x: n_tiles_model,
            },
        );

        let pass_b_kernel = &self.pass_b[dtype_idx];
        let encoder_b = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        pass_b_kernel.encode(
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
