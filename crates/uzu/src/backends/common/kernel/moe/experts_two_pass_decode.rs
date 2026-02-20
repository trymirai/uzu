use super::{
    MoeExpertsTwoPassArguments, MoePassARowMapArguments, MoePassATileBuildArguments, MoePassATileCountsArguments,
    MoePassATileDispatchArguments, MoePassATileKernels, MoePassATileScanArguments,
};
use crate::{
    DataType,
    backends::common::{
        CommandBuffer,
        kernel::{Backend, Kernels, MoeExpertsDecodeDownFused2DKernel, MoeExpertsDecodePassAKernel},
    },
};

static DTYPES: [DataType; 3] = [DataType::F16, DataType::BF16, DataType::F32];

pub struct MoeExpertsTwoPassDecodeBlock<B: Backend> {
    pass_a_tile: MoePassATileKernels<B>,
    pass_a_indirect: Vec<Vec<<B::Kernels as Kernels>::MoeExpertsDecodePassAKernel>>,
    fused_down: Vec<<B::Kernels as Kernels>::MoeExpertsDecodeDownFused2DKernel>,
}

impl<B: Backend> MoeExpertsTwoPassDecodeBlock<B> {
    pub fn new(ctx: &B::Context) -> Result<Self, B::Error> {
        let mut pass_a_indirect = vec![];
        for gate in 0u32..4u32 {
            let mut kernels = vec![];
            for dtype in &DTYPES {
                let kernel = <B::Kernels as Kernels>::MoeExpertsDecodePassAKernel::new(ctx, *dtype, gate)?;
                kernels.push(kernel);
            }
            pass_a_indirect.push(kernels);
        }

        let mut fused_down = Vec::with_capacity(DTYPES.len());
        for dtype in &DTYPES {
            let kernel = <B::Kernels as Kernels>::MoeExpertsDecodeDownFused2DKernel::new(ctx, *dtype, DataType::F32)?;
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
        command_buffer: &B::CommandBuffer,
        args: &MoeExpertsTwoPassArguments<B>,
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

        let gate_idx = args.gating_code.min(3) as usize;
        let dtype_idx = DTYPES.iter().position(|dtype| *dtype == args.data_type).unwrap();

        // pass a
        command_buffer.with_compute_encoder(|encoder| {
            let pass_a_kernel = &self.pass_a_indirect[gate_idx][dtype_idx];
            pass_a_kernel.encode(
                args.x_perm_buffer,
                args.expert_offsets,
                args.w13_all,
                args.hidden_buffer,
                args.up_biases,
                args.d_model as u32,
                args.d_ff as u32,
                args.e as u32,
                args.gate_clip_min,
                args.gate_clip_max,
                args.up_clip_min,
                args.up_clip_max,
                args.silu_alpha,
                args.tile_map,
                args.dispatch_args,
                encoder,
            );
        });

        // pass b
        command_buffer.with_compute_encoder(|encoder| {
            let pass_b_kernel = &self.fused_down[dtype_idx];
            pass_b_kernel.encode(
                args.hidden_buffer,
                args.row_expert_map,
                args.w2_all,
                args.down_biases,
                args.output_buffer,
                args.total_rows as u32,
                args.d_model as u32,
                args.d_ff as u32,
                args.e as u32,
                encoder,
            );
        });
    }
}
