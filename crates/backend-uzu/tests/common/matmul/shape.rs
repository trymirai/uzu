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

const BENCH_NK: &[(usize, usize)] = &[(2048, 2048), (2048, 4096), (4096, 4096), (4096, 14336), (14336, 4096), (14336, 14336)];

pub fn bench_quant_gemm_shapes(bits: u32) -> impl Iterator<Item = Shape> {
    let block_size: usize = if bits == 4 { 512 } else { 256 };
    let ms = &[4usize, 5, 6, 7, 8, 16, 32, 48, 64];
    BENCH_NK
        .iter()
        .filter(move |&&(n, k)| n % 32 == 0 && k % block_size == 0)
        .flat_map(move |&(n, k)| ms.iter().map(move |&m| Shape::new(m, k, n)))
}

pub fn bench_quant_gemv_shapes(bits: u32) -> impl Iterator<Item = Shape> {
    let block_size: usize = if bits == 4 { 512 } else { 256 };
    let nk: &[(usize, usize)] = &[(4096, 4096), (4096, 14336), (14336, 4096), (14336, 14336)];
    let ms = &[1usize, 2, 4, 8];
    nk.iter()
        .filter(move |&&(n, k)| n % 8 == 0 && k % block_size == 0)
        .flat_map(move |&(n, k)| ms.iter().map(move |&m| Shape::new(m, k, n)))
}
