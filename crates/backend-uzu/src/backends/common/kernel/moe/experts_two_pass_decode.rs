use super::{
    MoeExpertsTwoPassArguments, MoePassARowMapArguments, MoePassATileBuildArguments, MoePassATileCountsArguments,
    MoePassATileDispatchArguments, MoePassATileKernels, MoePassATileScanArguments,
};
use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{Kernels, MoeExpertsDecodeDownFused2DKernel, MoeExpertsDecodePassAKernel},
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
        encoder: &mut Encoder<B>,
        args: MoeExpertsTwoPassArguments<B>,
    ) -> Result<Allocation<B>, B::Error> {
        const BLOCK_M: u32 = 4;
        let h_blocks = (args.d_ff as u32 + BLOCK_M - 1) / BLOCK_M;
        let tile_counts = self.pass_a_tile.encode_counts(
            encoder,
            MoePassATileCountsArguments {
                expert_offsets: args.expert_offsets,
                e: args.e,
                h_blocks,
            },
        )?;

        let tile_scan = self.pass_a_tile.encode_scan(
            encoder,
            MoePassATileScanArguments {
                tile_counts: &tile_counts,
                e: args.e,
            },
        )?;

        let row_expert_map = self.pass_a_tile.encode_row_map(
            encoder,
            MoePassARowMapArguments {
                expert_offsets: args.expert_offsets,
                total_rows: args.total_rows,
                e: args.e,
            },
        )?;

        let tile_map = self.pass_a_tile.encode_build_map(
            encoder,
            MoePassATileBuildArguments {
                expert_offsets: args.expert_offsets,
                tile_offsets: &tile_scan.tile_offsets,
                row_expert_map: &row_expert_map,
                total_rows: args.total_rows,
                h_blocks,
            },
        )?;

        let dispatch_args = self.pass_a_tile.encode_dispatch_args(
            encoder,
            MoePassATileDispatchArguments {
                total_tiles: &tile_scan.total_tiles,
                num_tiles_y: 1,
            },
        )?;

        let gate_idx = args.gating_code.min(3) as usize;
        let dtype_idx = DTYPES.iter().position(|dtype| *dtype == args.data_type).unwrap();
        let mut hidden = encoder.allocate_scratch(size_for_shape(&[args.total_rows, args.d_ff], DataType::F32))?;

        let pass_a_kernel = &self.pass_a_indirect[gate_idx][dtype_idx];
        pass_a_kernel.encode(
            args.x_perm,
            args.expert_offsets,
            args.w13_all,
            &mut hidden,
            args.up_biases,
            args.d_model as u32,
            args.d_ff as u32,
            args.e as u32,
            args.gate_clip_min,
            args.gate_clip_max,
            args.up_clip_min,
            args.up_clip_max,
            args.silu_alpha,
            &tile_map,
            &dispatch_args,
            encoder,
        );

        let mut output = encoder.allocate_scratch(size_for_shape(&[args.total_rows, args.d_model], args.data_type))?;
        let pass_b_kernel = &self.fused_down[dtype_idx];
        pass_b_kernel.encode(
            &hidden,
            &row_expert_map,
            args.w2_all,
            args.down_biases,
            &mut output,
            args.total_rows as u32,
            args.d_model as u32,
            args.d_ff as u32,
            args.e as u32,
            encoder,
        );
        Ok(output)
    }
}
