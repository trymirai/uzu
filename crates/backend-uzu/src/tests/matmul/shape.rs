use derive_more::Display;

#[derive(Debug, Display, Clone, Copy, PartialEq, Eq)]
#[display("M[{m}]K[{k}]N[{n}]")]
pub struct Shape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

impl Shape {
    pub const fn new(
        m: usize,
        k: usize,
        n: usize,
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

const BENCH_NK: &[(usize, usize)] =
    &[(2048, 2048), (2048, 4096), (4096, 4096), (4096, 14336), (14336, 4096), (14336, 14336)];

pub fn bench_quant_gemm_shapes(bits: u32) -> impl Iterator<Item = Shape> {
    let block_size: usize = if bits == 4 {
        512
    } else {
        256
    };
    let ms = &[4usize, 5, 6, 7, 8, 16, 32, 48, 64];
    BENCH_NK
        .iter()
        .filter(move |&&(n, k)| n % 32 == 0 && k % block_size == 0)
        .flat_map(move |&(n, k)| ms.iter().map(move |&m| Shape::new(m, k, n)))
}

pub fn bench_quant_gemv_shapes(bits: u32) -> impl Iterator<Item = Shape> {
    let block_size: usize = if bits == 4 {
        512
    } else {
        256
    };
    let nk: &[(usize, usize)] = &[(4096, 4096), (4096, 14336), (14336, 4096), (14336, 14336)];
    let ms = &[1usize, 2, 4];
    nk.iter()
        .filter(move |&&(n, k)| n % 8 == 0 && k % block_size == 0)
        .flat_map(move |&(n, k)| ms.iter().map(move |&m| Shape::new(m, k, n)))
}

const QWEN3_LAYERS: &[(&str, usize, usize)] = &[
    ("0.8b_qkv", 1024, 3072),
    ("0.8b_o", 2048, 1024),
    ("0.8b_gate", 1024, 2048),
    ("0.8b_up", 1024, 7168),
    ("0.8b_down", 3584, 1024),
    ("0.8b_in", 1024, 8224),
    ("2b_qkv", 2048, 3072),
    ("2b_o", 2048, 2048),
    ("2b_up", 2048, 12288),
    ("2b_down", 6144, 2048),
    ("2b_in", 2048, 8224),
    ("4b_qkv", 2560, 6144),
    ("4b_o", 4096, 2560),
    ("4b_gate", 2560, 4096),
    ("4b_up", 2560, 18432),
    ("4b_down", 9216, 2560),
    ("4b_in", 2560, 12352),
];

pub fn qwen3_layer_shapes(bits: u32) -> impl Iterator<Item = (&'static str, Shape)> {
    let block_size: usize = if bits == 4 {
        512
    } else {
        256
    };
    let ms = &[1usize, 2, 4, 8, 16, 32, 64];
    QWEN3_LAYERS
        .iter()
        .filter(move |&&(_, k, _)| k.is_multiple_of(block_size))
        .flat_map(move |&(label, k, n)| ms.iter().map(move |&m| (label, Shape::new(m, k, n))))
}
