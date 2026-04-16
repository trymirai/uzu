use crate::{
    DataType,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        kernel::{MoeExpertsDecodeSinglePassAKernel, MoeExpertsDecodeSinglePassBKernel},
    },
};

static DTYPES: [DataType; 3] = [DataType::F16, DataType::BF16, DataType::F32];

/// Arguments for single-token MoE decode (T=1 optimized path)
pub struct MoeExpertsSingleDecodeArguments<'a, B: Backend> {
    /// Input activation [d_model]
    pub x: &'a Allocation<B>,
    /// Top-K expert indices from router [K]
    pub topk_ids: &'a Allocation<B>,
    /// Top-K probabilities from router [K]
    pub topk_probs: &'a Allocation<B>,
    /// Up/gate projection weights [E, 2*d_ff, d_model]
    pub w13_all: &'a B::Buffer,
    /// Down projection weights [E, d_model, d_ff]
    pub w2_all: &'a B::Buffer,
    /// Up/gate biases [E, 2*d_ff]
    pub up_biases: &'a B::Buffer,
    /// Down biases [E, d_model]
    pub down_biases: &'a B::Buffer,
    /// Hidden buffer [K, d_ff] - intermediate storage (f32)
    pub hidden: &'a mut Allocation<B>,
    /// Final output [d_model]
    pub y: &'a mut Allocation<B>,
    /// Model dimension
    pub d_model: usize,
    /// FFN hidden dimension
    pub d_ff: usize,
    /// Number of active experts per token
    pub k: usize,
    /// Gating activation: 0=GELU, 1=SiLU, 2=SwiGLU, 3=GEGLU
    pub gating_code: u32,
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
    /// Data type
    pub data_type: DataType,
}

pub struct MoeExpertsSingleDecodeKernels<B: Backend> {
    pass_a: Vec<Vec<<B::Kernels as Kernels>::MoeExpertsDecodeSinglePassAKernel>>,
    pass_b: Vec<<B::Kernels as Kernels>::MoeExpertsDecodeSinglePassBKernel>,
}

impl<B: Backend> MoeExpertsSingleDecodeKernels<B> {
    pub fn new(ctx: &B::Context) -> Result<Self, B::Error> {
        let mut pass_a = vec![];
        for gate in 0..4 {
            let mut kernels = vec![];
            for dtype in &DTYPES {
                let kernel =
                    <B::Kernels as Kernels>::MoeExpertsDecodeSinglePassAKernel::new(ctx, (*dtype).into(), gate)?;
                kernels.push(kernel)
            }
            pass_a.push(kernels);
        }

        let mut pass_b = Vec::with_capacity(DTYPES.len());
        for dtype in &DTYPES {
            pass_b.push(<B::Kernels as Kernels>::MoeExpertsDecodeSinglePassBKernel::new(ctx, (*dtype).into())?);
        }

        Ok(Self {
            pass_a,
            pass_b,
        })
    }

    pub fn encode(
        &self,
        encoder: &mut Encoder<B>,
        args: MoeExpertsSingleDecodeArguments<B>,
    ) {
        if args.k == 0 {
            return;
        }

        let gate_idx = args.gating_code.min(3) as usize;
        let dtype_idx = DTYPES.iter().position(|dtype| *dtype == args.data_type).unwrap();

        // Pass A: x @ W13[expert] -> hidden
        let kernel = &self.pass_a[gate_idx][dtype_idx];
        kernel.encode(
            args.x,
            args.topk_ids,
            args.w13_all,
            args.up_biases,
            &mut *args.hidden,
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
        let kernel = &self.pass_b[dtype_idx];
        kernel.encode(
            &*args.hidden,
            args.topk_ids,
            args.topk_probs,
            args.w2_all,
            args.down_biases,
            args.y,
            args.d_model as u32,
            args.d_ff as u32,
            args.k as u32,
            encoder,
        );
    }
}
