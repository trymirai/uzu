use metal::{MTLCommandBuffer, MTLCommandEncoder};
use objc2::{
    __framework_prelude::{ProtocolObject, Retained},
    Message,
};

use crate::{
    DataType,
    backends::{
        common::kernel::{
            MoeExpertsDecodeDownFused2DKernel, MoeExpertsDecodePassAKernel,
        },
        metal::{
            MTLContext, MTLError,
            kernel::{
                MoeExpertsTwoPassArguments,
                dsl::{
                    MoeExpertsDecodeDownFused2DMetalKernel,
                    MoeExpertsDecodePassAMetalKernel,
                },
                moe::{
                    MoePassARowMapArguments, MoePassATileBuildArguments,
                    MoePassATileCountsArguments, MoePassATileDispatchArguments,
                    MoePassATileKernels, MoePassATileScanArguments,
                },
            },
        },
    },
};

static DTYPES: [DataType; 3] = [DataType::F16, DataType::BF16, DataType::F32];

pub struct MoeExpertsTwoPassDecodeKernels {
    pass_a_tile: MoePassATileKernels,
    pass_a_indirect: Vec<Vec<MoeExpertsDecodePassAMetalKernel>>,
    fused_down: Vec<MoeExpertsDecodeDownFused2DMetalKernel>,
}

impl MoeExpertsTwoPassDecodeKernels {
    pub fn new(ctx: &MTLContext) -> Result<Self, MTLError> {
        let mut pass_a_indirect = vec![];
        for gate in 0u32..4u32 {
            let mut kernels = vec![];
            for dtype in &DTYPES {
                let kernel =
                    MoeExpertsDecodePassAMetalKernel::new(ctx, *dtype, gate)?;
                kernels.push(kernel);
            }
            pass_a_indirect.push(kernels);
        }

        let mut fused_down = Vec::with_capacity(DTYPES.len());
        for dtype in &DTYPES {
            let kernel = MoeExpertsDecodeDownFused2DMetalKernel::new(
                ctx,
                *dtype,
                DataType::F32,
            )?;
            fused_down.push(kernel);
        }

        Ok(Self {
            pass_a_tile: MoePassATileKernels::new(ctx)?,
            pass_a_indirect,
            fused_down,
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

        // pass a tile
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
        );
        self.pass_a_tile.encode_scan(
            command_buffer,
            &MoePassATileScanArguments {
                tile_counts: args.tile_counts,
                tile_offsets: args.tile_offsets,
                total_tiles: args.total_tiles,
                e: args.e,
            },
        );
        self.pass_a_tile.encode_row_map(
            command_buffer,
            &MoePassARowMapArguments {
                expert_offsets: args.expert_offsets,
                row_expert_map: args.row_expert_map,
                total_rows: args.total_rows,
                e: args.e,
            },
        );
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
        );
        self.pass_a_tile.encode_dispatch_args(
            command_buffer,
            &MoePassATileDispatchArguments {
                total_tiles: args.total_tiles,
                dispatch_args: args.dispatch_args,
                num_tiles_y: 1,
            },
        );

        // pass a
        let gate_idx = args.gating_code.min(3) as usize;
        let dtype_idx =
            DTYPES.iter().position(|&t| t == args.data_type.into()).unwrap();
        let pass_a_encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        let pass_a_kernel = &self.pass_a_indirect[gate_idx][dtype_idx];
        pass_a_kernel.encode(
            &args.x_perm_buffer.retain(),
            &args.expert_offsets.retain(),
            &args.w13_all.retain(),
            &args.hidden_buffer.retain(),
            &args.up_biases.retain(),
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
            &pass_a_encoder,
        );
        pass_a_encoder.end_encoding();

        // pass b
        let pass_b_kernel = &self.fused_down[dtype_idx];
        let pass_b_encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        pass_b_kernel.encode(
            &args.hidden_buffer.retain(),
            &args.row_expert_map.retain(),
            &args.w2_all.retain(),
            &args.down_biases.retain(),
            &args.output_buffer.retain(),
            args.total_rows as u32,
            args.d_model as u32,
            args.d_ff as u32,
            args.e as u32,
            &pass_b_encoder,
        );
        pass_b_encoder.end_encoding();
    }
}
