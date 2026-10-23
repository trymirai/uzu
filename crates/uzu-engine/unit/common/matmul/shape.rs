use derive_more::Display;

use crate::backends::common::{
    gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
    kernel::{
        ActivationQuantization,
        matmul::{ActivationFormat, Int8CodeLayout, MatmulShape, QuantParamsLayout},
    },
};

#[derive(Debug, Display, Clone, Copy, PartialEq, Eq)]
#[display("M[{m}]K[{k}]N[{n}]")]
pub struct Shape {
    pub m: u32,
    pub k: u32,
    pub n: u32,
}

impl Shape {
    pub const fn new(
        m: u32,
        k: u32,
        n: u32,
    ) -> Self {
        Self {
            m,
            k,
            n,
        }
    }
}

const SHAPES_TINY: &[Shape] = &[Shape::new(64, 64, 64), Shape::new(16, 128, 256), Shape::new(128, 256, 128)];

const SHAPES_UNALIGNED: &[Shape] = &[
    Shape::new(7, 33, 11),
    Shape::new(33, 2048, 2048),
    Shape::new(64, 2048, 33),
    Shape::new(200, 2048, 2048),
    Shape::new(128, 2048, 200),
];

const SHAPES_MEDIUM: &[Shape] = &[Shape::new(128, 2048, 2048), Shape::new(256, 4096, 4096)];

pub fn all_correctness_shapes() -> impl Iterator<Item = Shape> {
    SHAPES_TINY.iter().chain(SHAPES_UNALIGNED.iter()).chain(SHAPES_MEDIUM.iter()).copied()
}

const BENCH_FP_GEMM: &[Shape] = &[
    Shape::new(128, 2048, 8192),
    Shape::new(128, 4096, 14336),
    Shape::new(256, 4096, 4096),
    Shape::new(512, 8192, 2048),
];

pub fn bench_fp_gemm_shapes() -> impl Iterator<Item = Shape> {
    BENCH_FP_GEMM.iter().copied()
}

const BENCH_NK: &[(u32, u32)] =
    &[(2048, 2048), (2048, 4096), (4096, 4096), (4096, 14336), (14336, 4096), (14336, 14336)];

pub fn bench_quant_gemm_shapes(bits: u32) -> impl Iterator<Item = Shape> {
    let block_size: u32 = if bits == 4 {
        512
    } else {
        256
    };
    // The tile policy only differs for m that is not a multiple of 8, so keep 4..16 contiguous.
    let ms = &[4u32, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 48, 64];
    BENCH_NK
        .iter()
        .filter(move |&&(n, k)| n % 32 == 0 && k % block_size == 0)
        .flat_map(move |&(n, k)| ms.iter().map(move |&m| Shape::new(m, k, n)))
}

pub fn bench_quant_gemv_shapes(bits: u32) -> impl Iterator<Item = Shape> {
    let block_size: u32 = if bits == 4 {
        512
    } else {
        256
    };
    let nk: &[(u32, u32)] = &[(4096, 4096), (4096, 14336), (14336, 4096), (14336, 14336)];
    let ms = &[1u32, 2, 4];
    nk.iter()
        .filter(move |&&(n, k)| n % 8 == 0 && k % block_size == 0)
        .flat_map(move |&(n, k)| ms.iter().map(move |&m| Shape::new(m, k, n)))
}

/// Checkpoint products by label, K and N, as the 0.8B, 4B and 27B checkpoint headers record them. Tied readouts multiply
/// by the embedding table; the 27B readout is its own linear.
const QWEN3_LAYERS: &[(&str, u32, u32)] = &[
    ("0.8b_qkvg", 1024, 5120),
    ("0.8b_o", 2048, 1024),
    ("0.8b_up", 1024, 7168),
    ("0.8b_down", 3584, 1024),
    ("0.8b_in", 1024, 8224),
    ("0.8b_tied_readout", 1024, 248320),
    ("2b_qkv", 2048, 3072),
    ("2b_o", 2048, 2048),
    ("2b_up", 2048, 12288),
    ("2b_down", 6144, 2048),
    ("2b_in", 2048, 8224),
    ("4b_qkvg", 2560, 10240),
    ("4b_o", 4096, 2560),
    ("4b_up", 2560, 18432),
    ("4b_down", 9216, 2560),
    ("4b_in", 2560, 12352),
    ("4b_tied_readout", 2560, 248320),
    ("27b_qkvg", 5120, 14336),
    ("27b_o", 6144, 5120),
    ("27b_gate_up", 5120, 34816),
    ("27b_down", 17408, 5120),
    ("27b_in", 5120, 16480),
    ("27b_readout", 5120, 248320),
];

pub fn qwen3_layer_shapes(bits: u32) -> impl Iterator<Item = (&'static str, Shape)> {
    let block_size: u32 = if bits == 4 {
        512
    } else {
        256
    };
    // Measured requests use 12/395 rows; the 27B verifier/readout also use 15/16.
    let ms = &[1u32, 2, 4, 8, 12, 15, 16, 32, 64, 128, 256, 395, 512];
    QWEN3_LAYERS
        .iter()
        .filter(move |&&(_, k, _)| k.is_multiple_of(block_size))
        .flat_map(move |&(label, k, n)| ms.iter().map(move |&m| (label, Shape::new(m, k, n))))
}

pub fn matmul_shape(
    m: u32,
    n: u32,
    k: u32,
) -> MatmulShape {
    MatmulShape {
        m,
        n,
        k,
        b_transpose: true,
        b_leading_dimension: None,
        b_prologue: GemmBPrologueKind::FullPrecision,
        b_is_trellis: false,
        b_bits: None,
        b_group_size: None,
        signed_codes: false,
        a: ActivationFormat::Bf16,
        gathered: false,
        params_layout: Some(QuantParamsLayout::OutputGroup),
        d_transform: GemmDTransform::empty(),
    }
}

pub fn quantized_matmul_shape(
    m: u32,
    n: u32,
    k: u32,
    bits: u32,
    group: u32,
) -> MatmulShape {
    MatmulShape {
        b_prologue: if bits == 8 {
            GemmBPrologueKind::ScaleSymmetricDequant
        } else {
            GemmBPrologueKind::ScaleZeroPointDequant
        },
        b_bits: Some(bits),
        b_group_size: Some(group),
        signed_codes: bits == 8,
        params_layout: Some(QuantParamsLayout::GroupOutput),
        ..matmul_shape(m, n, k)
    }
}

pub fn a8_format() -> ActivationFormat {
    ActivationFormat::Int8(ActivationQuantization::new(128, 128, false, Int8CodeLayout::GroupedByNibble).unwrap())
}
