use crate::backends::common::kernel::{MoeExpertsDecodeSinglePassAKernel as _, MoeExpertsDecodeSinglePassBKernel as _};
use crate::backends::metal::kernel::dsl::{
    MoeExpertsDecodeSinglePassAMetalKernel, MoeExpertsDecodeSinglePassBMetalKernel,
};
use crate::backends::metal::{KernelDataType, MTLContext, MTLError};
use metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder};
use objc2::__framework_prelude::ProtocolObject;
use objc2::Message;
use objc2::rc::Retained;

static DTYPES: [KernelDataType; 3] = [KernelDataType::Float16, KernelDataType::BFloat16, KernelDataType::Float32];

/// Arguments for single-token MoE decode (T=1 optimized path)
#[derive(Debug)]
pub struct MoeExpertsSingleDecodeArguments<'a> {
    /// Input activation [d_model]
    pub x: &'a ProtocolObject<dyn MTLBuffer>,
    /// Top-K expert indices from router [K]
    pub topk_ids: &'a ProtocolObject<dyn MTLBuffer>,
    /// Top-K probabilities from router [K]
    pub topk_probs: &'a ProtocolObject<dyn MTLBuffer>,
    /// Up/gate projection weights [E, 2*d_ff, d_model]
    pub w13_all: &'a ProtocolObject<dyn MTLBuffer>,
    /// Down projection weights [E, d_model, d_ff]
    pub w2_all: &'a ProtocolObject<dyn MTLBuffer>,
    /// Up/gate biases [E, 2*d_ff]
    pub up_biases: &'a ProtocolObject<dyn MTLBuffer>,
    /// Down biases [E, d_model]
    pub down_biases: &'a ProtocolObject<dyn MTLBuffer>,
    /// Hidden buffer [K, d_ff] - intermediate storage (f32)
    pub hidden: &'a ProtocolObject<dyn MTLBuffer>,
    /// Final output [d_model]
    pub y: &'a ProtocolObject<dyn MTLBuffer>,
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
    pub data_type: KernelDataType,
}

pub struct MoeExpertsSingleDecodeKernels {
    pass_a: Vec<Vec<MoeExpertsDecodeSinglePassAMetalKernel>>,
    pass_b: Vec<MoeExpertsDecodeSinglePassBMetalKernel>,
}

impl MoeExpertsSingleDecodeKernels {
    pub fn new(ctx: &MTLContext) -> Result<Self, MTLError> {
        let mut pass_a = vec![];
        for gate in 0..4 {
            let mut kernels = vec![];
            for dtype in &DTYPES {
                let kernel = MoeExpertsDecodeSinglePassAMetalKernel::new(ctx, (*dtype).into(), gate)?;
                kernels.push(kernel)
            }
            pass_a.push(kernels);
        }

        let mut pass_b = Vec::with_capacity(DTYPES.len());
        for dtype in &DTYPES {
            pass_b.push(MoeExpertsDecodeSinglePassBMetalKernel::new(ctx, (*dtype).into())?);
        }

        Ok(Self {
            pass_a,
            pass_b,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        args: MoeExpertsSingleDecodeArguments,
    ) {
        if args.k == 0 {
            return;
        }

        let gate_idx = args.gating_code.min(3) as usize;
        let dtype_idx = DTYPES.iter().position(|dtype| *dtype == args.data_type).unwrap();

        // Pass A: x @ W13[expert] -> hidden
        {
            let kernel = &self.pass_a[gate_idx][dtype_idx];
            let encoder =
                command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
            kernel.encode(
                &args.x.retain(),
                &args.topk_ids.retain(),
                &args.w13_all.retain(),
                &args.up_biases.retain(),
                &args.hidden.retain(),
                args.d_model as u32,
                args.d_ff as u32,
                args.k as u32,
                args.silu_alpha,
                args.gate_clip_min,
                args.gate_clip_max,
                args.up_clip_min,
                args.up_clip_max,
                &encoder,
            );
            encoder.end_encoding();
        }

        // Pass B: 8 simdgroups (256 threads), outputs final y directly
        {
            let kernel = &self.pass_b[dtype_idx];
            let encoder =
                command_buffer.new_compute_command_encoder().expect("Failed to create compute command encoder");
            kernel.encode(
                &args.hidden.retain(),
                &args.topk_ids.retain(),
                &args.topk_probs.retain(),
                &args.w2_all.retain(),
                &args.down_biases.retain(),
                &args.y.retain(),
                args.d_model as u32,
                args.d_ff as u32,
                args.k as u32,
                &encoder,
            );
            encoder.end_encoding();
        }
    }
}
