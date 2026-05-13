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
        Self { m, k, n }
    }

    pub const fn flops(&self) -> usize {
        2 * self.m * self.k * self.n
    }
}

pub const SHAPES_TINY: &[Shape] = &[Shape::new(64, 64, 64), Shape::new(16, 128, 256), Shape::new(128, 256, 128)];

pub const SHAPES_UNALIGNED: &[Shape] = &[
    Shape::new(7, 33, 11),
    Shape::new(33, 2048, 2048),
    Shape::new(64, 2048, 33),
    Shape::new(200, 2048, 2048),
    Shape::new(128, 2048, 200),
];

pub const SHAPES_MEDIUM: &[Shape] = &[Shape::new(128, 2048, 2048), Shape::new(256, 4096, 4096)];

pub const SHAPES_BENCH: &[Shape] =
    &[Shape::new(128, 2048, 8192), Shape::new(128, 4096, 14336), Shape::new(256, 4096, 4096), Shape::new(512, 8192, 2048)];

pub fn all_correctness_shapes() -> impl Iterator<Item = Shape> {
    SHAPES_TINY.iter().chain(SHAPES_UNALIGNED.iter()).chain(SHAPES_MEDIUM.iter()).copied()
}
