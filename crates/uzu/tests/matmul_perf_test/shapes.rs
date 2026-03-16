pub struct TestShape {
    pub batch: usize,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl std::fmt::Display for TestShape {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}x{}x{}", self.batch, self.input_dim, self.output_dim)
    }
}

const BATCH_SIZES: &[usize] = &[1, 4, 16, 64, 128, 256, 512];

const MODEL_DIMS: &[(usize, usize)] = &[
    (896, 896),
    (896, 4864),
    (4864, 896),
    (2048, 2048),
    (2048, 8192),
    (8192, 2048),
    (4096, 4096),
    (4096, 14336),
    (14336, 4096),
];

pub fn test_shapes() -> Vec<TestShape> {
    MODEL_DIMS
        .iter()
        .flat_map(|&(input_dim, output_dim)| {
            BATCH_SIZES.iter().map(move |&batch| TestShape {
                batch,
                input_dim,
                output_dim,
            })
        })
        .collect()
}
