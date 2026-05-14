// TODO(LuckyIYI): figure out if this was made dead code by accident and restore or delete

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        kernel::{MoeExpertsDecodeSinglePassAKernel, MoeExpertsDecodeSinglePassBKernel},
    },
};

/// Arguments for single-token MoE decode (T=1 optimized path)
pub struct MoeExpertsSingleDecodeArguments<'a, B: Backend> {
    /// Input activation [d_model]
    pub x: &'a Allocation<B>,
    /// Top-K expert indices from router [K]
    pub topk_ids: &'a Allocation<B>,
    /// Top-K probabilities from router [K]
    pub topk_probs: &'a Allocation<B>,
    /// Up/gate projection weights [E, 2*d_ff, d_model]
    pub w13_all: &'a Allocation<B>,
    /// Down projection weights [E, d_model, d_ff]
    pub w2_all: &'a Allocation<B>,
    /// Up/gate biases [E, 2*d_ff]
    pub up_biases: &'a Allocation<B>,
    /// Down biases [E, d_model]
    pub down_biases: &'a Allocation<B>,
    /// Model dimension
    pub d_model: usize,
    /// FFN hidden dimension
    pub d_ff: usize,
    /// Number of active experts per token
    pub k: usize,
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
}

pub struct MoeExpertsSingleDecodeKernels<B: Backend> {
    pass_a: <B::Kernels as Kernels>::MoeExpertsDecodeSinglePassAKernel,
    pass_b: <B::Kernels as Kernels>::MoeExpertsDecodeSinglePassBKernel,
    data_type: DataType,
}

impl<B: Backend> MoeExpertsSingleDecodeKernels<B> {
    pub fn new(
        ctx: &B::Context,
        data_type: DataType,
        gating_code: u32,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            pass_a: <B::Kernels as Kernels>::MoeExpertsDecodeSinglePassAKernel::new(ctx, data_type, gating_code)?,
            pass_b: <B::Kernels as Kernels>::MoeExpertsDecodeSinglePassBKernel::new(ctx, data_type)?,
            data_type,
        })
    }

    pub fn encode(
        &self,
        args: MoeExpertsSingleDecodeArguments<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        // Pass A: x @ W13[expert] -> hidden
        let mut hidden = encoder.allocate_scratch(size_for_shape(&[args.k, args.d_ff], DataType::F32))?;
        self.pass_a.encode(
            args.x,
            args.topk_ids,
            args.w13_all,
            args.up_biases,
            &mut hidden,
            args.d_model as u32,
            args.d_ff as u32,
            args.k as u32,
            args.silu_alpha,
            args.gate_clip_min,
            args.gate_clip_max,
            args.up_clip_min,
            args.up_clip_max,
            encoder,
        );

        // Pass B: 8 simdgroups (256 threads), outputs final y directly
        let mut output = encoder.allocate_scratch(size_for_shape(&[args.d_model], self.data_type))?;
        self.pass_b.encode(
            &hidden,
            args.topk_ids,
            args.topk_probs,
            args.w2_all,
            args.down_biases,
            &mut output,
            args.d_model as u32,
            args.d_ff as u32,
            args.k as u32,
            encoder,
        );

        Ok(output)
    }
}
