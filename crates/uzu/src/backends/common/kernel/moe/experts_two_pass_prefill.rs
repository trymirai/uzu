use std::ops::{Deref, DerefMut};

use super::{
    MoeTileCountsArguments, MoeTileDispatchArguments, MoeTileMapBuildArguments, MoeTileMapKernels, MoeTileScanArguments,
};
use crate::{
    DataType,
    backends::common::{
        Backend, Encoder, Kernels,
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
        encoder: &mut Encoder<B>,
        mut args: MoeExpertsTwoPassArguments<B>,
    ) {
        if args.total_rows == 0 {
            return;
        }

        let hidden_bytes = args.total_rows * args.d_ff * args.data_type.size_in_bytes();
        encoder.encode_fill(args.hidden_buffer.deref_mut(), 0..hidden_bytes, 0);

        self.tile_map.encode_counts(
            encoder,
            MoeTileCountsArguments {
                offsets_buffer: args.expert_offsets,
                tile_counts_buffer: args.tile_counts.deref_mut(),
                e: args.e,
            },
        );
        self.tile_map.encode_scan(
            encoder,
            MoeTileScanArguments {
                tile_counts_buffer: args.tile_counts,
                tile_offsets_buffer: args.tile_offsets.deref_mut(),
                total_tiles_buffer: args.total_tiles.deref_mut(),
                e: args.e,
            },
        );
        self.tile_map.encode_build_map(
            encoder,
            MoeTileMapBuildArguments {
                expert_offsets: args.expert_offsets,
                tile_offsets: args.tile_offsets,
                tile_counts: args.tile_counts,
                tile_map: args.tile_map.deref_mut(),
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
            encoder,
            MoeTileDispatchArguments {
                total_tiles: args.total_tiles,
                dispatch_args: args.dispatch_args.deref_mut(),
                num_tiles_x: n_tiles_ff,
            },
        );

        let gate_idx = args.gating_code.min(3) as usize;
        let dtype_idx = DTYPES.iter().position(|t| *t == args.data_type).unwrap();

        let kernel_pass_a = &self.pass_a_indirect[gate_idx][dtype_idx];
        kernel_pass_a.encode(
            args.x_perm_buffer,
            args.expert_offsets,
            args.w13_all,
            args.up_biases,
            args.hidden_buffer.deref_mut(),
            args.d_model as u32,
            args.d_ff as u32,
            args.e as u32,
            args.gate_clip_min,
            args.gate_clip_max,
            args.up_clip_min,
            args.up_clip_max,
            args.silu_alpha,
            args.tile_map.deref(),
            args.dispatch_args.deref(),
            encoder,
        );

        let dispatch_args = MoeTileDispatchArguments {
            total_tiles: args.total_tiles,
            dispatch_args: args.dispatch_args.deref_mut(),
            num_tiles_x: n_tiles_model,
        };
        self.tile_map.encode_dispatch_args(encoder, dispatch_args);

        let kernel_pass_b = &self.pass_b_indirect[dtype_idx];
        kernel_pass_b.encode(
            args.hidden_buffer.deref(),
            args.expert_offsets,
            args.w2_all,
            args.down_biases,
            args.output_buffer.deref_mut(),
            args.d_model as u32,
            args.d_ff as u32,
            args.e as u32,
            args.tile_map.deref(),
            args.dispatch_args.deref(),
            encoder,
        );
    }
}

#[derive(Debug)]
pub struct MoeExpertsTwoPassArguments<'a, B: Backend> {
    pub x_perm_buffer: &'a B::Buffer,
    pub expert_offsets: &'a B::Buffer,
    pub row_expert_map: &'a mut B::Buffer,
    pub hidden_buffer: &'a mut B::Buffer,
    pub output_buffer: &'a mut B::Buffer,
    pub w13_all: &'a B::Buffer,
    pub w2_all: &'a B::Buffer,
    pub up_biases: &'a B::Buffer,
    pub down_biases: &'a B::Buffer,
    pub tile_counts: &'a mut B::Buffer,
    pub tile_offsets: &'a mut B::Buffer,
    pub tile_map: &'a mut B::Buffer,
    pub total_tiles: &'a mut B::Buffer,
    pub dispatch_args: &'a mut B::Buffer,
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
