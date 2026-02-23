use super::{
    MoeTileCountsArguments, MoeTileDispatchArguments, MoeTileMapBuildArguments, MoeTileMapKernels, MoeTileScanArguments,
};
use crate::{
    DataType,
    backends::common::{
        Backend, CommandBuffer, CopyEncoder, Kernels,
        kernel::{MoeExpertsPrefillPassAKernel, MoeExpertsPrefillPassBKernel},
    },
};

const DTYPES: [DataType; 3] = [DataType::F16, DataType::BF16, DataType::F32];

pub struct MoeExpertsTwoPassPrefillBlock<B: Backend> {
    tile_map: MoeTileMapKernels<B>,
    pass_a_indirect: Vec<Vec<<B::Kernels as Kernels>::MoeExpertsPrefillPassAKernel>>, // [gate][dtype]
    pass_b_indirect: Vec<<B::Kernels as Kernels>::MoeExpertsPrefillPassBKernel>,      // [dtype]
}

impl<B: Backend> MoeExpertsTwoPassPrefillBlock<B> {
    pub fn new(ctx: &B::Context) -> Result<Self, B::Error> {
        let dtypes = [DataType::F16, DataType::BF16, DataType::F32];

        let mut pass_a_indirect = vec![];
        for gate in 0u32..4u32 {
            let mut kernels = vec![];
            for dtype in dtypes {
                let kernel = <B::Kernels as Kernels>::MoeExpertsPrefillPassAKernel::new(ctx, dtype, gate)?;
                kernels.push(kernel);
            }
            pass_a_indirect.push(kernels);
        }

        let mut pass_b_indirect = vec![];
        for dtype in dtypes {
            let kernel = <B::Kernels as Kernels>::MoeExpertsPrefillPassBKernel::new(ctx, dtype)?;
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
        command_buffer: &B::CommandBuffer,
        args: &MoeExpertsTwoPassArguments<B>,
    ) {
        if args.total_rows == 0 {
            return;
        }

        let hidden_bytes = args.total_rows * args.d_ff * args.data_type.size_in_bytes();
        command_buffer.with_copy_encoder(|encoder| encoder.encode_fill(args.hidden_buffer, 0..hidden_bytes, 0));

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
        const COL_TILE_FF: u32 = 32; // Must match PASSA_BN in kernel
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

        let gate_idx = args.gating_code.min(3) as usize;
        let dtype_idx = DTYPES.iter().position(|t| *t == args.data_type).unwrap();

        command_buffer.with_compute_encoder(|encoder| {
            let kernel_pass_a = &self.pass_a_indirect[gate_idx][dtype_idx];
            kernel_pass_a.encode(
                args.x_perm_buffer,
                args.expert_offsets,
                args.w13_all,
                args.up_biases,
                args.hidden_buffer,
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

        let dispatch_args = MoeTileDispatchArguments {
            total_tiles: args.total_tiles,
            dispatch_args: args.dispatch_args,
            num_tiles_x: n_tiles_model,
        };
        self.tile_map.encode_dispatch_args(command_buffer, &dispatch_args);

        command_buffer.with_compute_encoder(|encoder| {
            let kernel_pass_b = &self.pass_b_indirect[dtype_idx];
            kernel_pass_b.encode(
                args.hidden_buffer,
                args.expert_offsets,
                args.w2_all,
                args.down_biases,
                args.output_buffer,
                args.d_model as u32,
                args.d_ff as u32,
                args.e as u32,
                args.tile_map,
                args.dispatch_args,
                encoder,
            );
        });
    }
}

#[derive(Debug)]
pub struct MoeExpertsTwoPassArguments<'a, B: Backend> {
    pub x_perm_buffer: &'a B::NativeBuffer,
    pub expert_offsets: &'a B::NativeBuffer,
    pub row_expert_map: &'a B::NativeBuffer,
    pub hidden_buffer: &'a B::NativeBuffer,
    pub output_buffer: &'a B::NativeBuffer,
    pub w13_all: &'a B::NativeBuffer,
    pub w2_all: &'a B::NativeBuffer,
    pub up_biases: &'a B::NativeBuffer,
    pub down_biases: &'a B::NativeBuffer,
    pub tile_counts: &'a B::NativeBuffer,
    pub tile_offsets: &'a B::NativeBuffer,
    pub tile_map: &'a B::NativeBuffer,
    pub total_tiles: &'a B::NativeBuffer,
    pub dispatch_args: &'a B::NativeBuffer,
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
