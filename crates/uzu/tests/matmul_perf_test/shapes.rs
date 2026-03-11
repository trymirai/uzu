use super::common::matmul::TestShape;

const BATCH_SIZES: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

const MODEL_DIMS: &[(usize, usize)] = &[
    (896, 896),
    (896, 4864),
    (4864, 896),
    (1024, 1024),
    (1024, 4096),
    (4096, 1024),
    (1152, 1152),
    (1152, 6912),
    (6912, 1152),
    (1536, 1536),
    (1536, 8960),
    (8960, 1536),
    (2048, 2048),
    (2048, 8192),
    (8192, 2048),
    (2560, 2560),
    (2560, 10240),
    (10240, 2560),
    (3072, 3072),
    (3072, 8192),
    (8192, 3072),
    (3584, 3584),
    (3584, 18944),
    (18944, 3584),
    (4096, 4096),
    (4096, 14336),
    (14336, 4096),
    (5120, 5120),
    (5120, 17408),
    (17408, 5120),
];

pub fn test_shapes() -> Vec<TestShape> {
    let grid_shapes = BATCH_SIZES.iter().flat_map(|&batch| {
        [512, 1024, 2048].iter().flat_map(move |&input_dim| {
            [512, 1024, 2048].iter().map(move |&output_dim| TestShape {
                batch,
                input_dim,
                output_dim,
            })
        })
    });

    let model_shapes = MODEL_DIMS.iter().flat_map(|&(input_dim, output_dim)| {
        BATCH_SIZES.iter().map(move |&batch| TestShape {
            batch,
            input_dim,
            output_dim,
        })
    });

    grid_shapes.chain(model_shapes).collect()
}
