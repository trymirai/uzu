use super::MoeTileMapKernels;
use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
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
        args: MoeExpertsTwoPassArguments<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let tile_counts = self.tile_map.encode_counts(encoder, args.expert_offsets, args.e)?;

        let tile_scan = self.tile_map.encode_scan(encoder, &tile_counts, args.e)?;

        let tile_map = self.tile_map.encode_build_map(
            encoder,
            args.expert_offsets,
            &tile_scan.tile_offsets,
            &tile_counts,
            args.total_rows,
            args.e,
        )?;
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
        let pass_a_dispatch_args = self.tile_map.encode_dispatch_args(encoder, &tile_scan.total_tiles, n_tiles_ff)?;

        let gate_idx = args.gating_code.min(3) as usize;
        let dtype_idx = DTYPES.iter().position(|t| *t == args.data_type).unwrap();
        let mut hidden = encoder.allocate_scratch(size_for_shape(&[args.total_rows, args.d_ff], DataType::F32))?;
        encoder.encode_fill(&mut hidden, 0);

        let kernel_pass_a = &self.pass_a_indirect[gate_idx][dtype_idx];
        kernel_pass_a.encode(
            args.x_perm,
            args.expert_offsets,
            args.w13_all,
            args.up_biases,
            &mut hidden,
            args.d_model as u32,
            args.d_ff as u32,
            args.e as u32,
            args.gate_clip_min,
            args.gate_clip_max,
            args.up_clip_min,
            args.up_clip_max,
            args.silu_alpha,
            &tile_map,
            &pass_a_dispatch_args,
            encoder,
        );

        let pass_b_dispatch_args =
            self.tile_map.encode_dispatch_args(encoder, &tile_scan.total_tiles, n_tiles_model)?;

        let mut output = encoder.allocate_scratch(size_for_shape(&[args.total_rows, args.d_model], args.data_type))?;
        let kernel_pass_b = &self.pass_b_indirect[dtype_idx];
        kernel_pass_b.encode(
            &hidden,
            args.expert_offsets,
            args.w2_all,
            args.down_biases,
            &mut output,
            args.d_model as u32,
            args.d_ff as u32,
            args.e as u32,
            &tile_map,
            &pass_b_dispatch_args,
            encoder,
        );
        Ok(output)
    }
}

pub struct MoeExpertsTwoPassArguments<'a, B: Backend> {
    pub x_perm: &'a Allocation<B>,
    pub expert_offsets: &'a Allocation<B>,
    pub w13_all: &'a Allocation<B>,
    pub w2_all: &'a Allocation<B>,
    pub up_biases: &'a Allocation<B>,
    pub down_biases: &'a Allocation<B>,
    pub total_rows: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub e: usize,
    pub gating_code: u32,
    pub gate_clip_min: f32,
    pub gate_clip_max: f32,
    pub up_clip_min: f32,
    pub up_clip_max: f32,
    pub silu_alpha: f32,
    pub data_type: DataType,
}
