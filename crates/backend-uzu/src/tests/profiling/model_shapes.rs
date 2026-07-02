pub fn profiling_model_dimensions() -> [usize; 4] {
    [512, 2048, 4096, 8192]
}

pub const ATTENTION_NUM_HEADS: usize = 32;
pub const ATTENTION_NUM_GROUPS: usize = 8;
pub const ATTENTION_HEAD_DIM: usize = 64;

pub fn attention_context_lengths() -> [usize; 4] {
    [128, 512, 2048, 8192]
}

pub const DECODE_TOKEN_COUNT: usize = 1;

pub const REAL_TOKEN_COUNTS: [usize; 4] = [DECODE_TOKEN_COUNT, 128, 512, 2048];

pub const VOCABULARY_SIZE: usize = 128256;
pub const PROFILING_VOCABULARY_SIZE: usize = 8192;

pub fn profiling_vocabulary_sizes() -> [usize; 3] {
    [32768, 65536, VOCABULARY_SIZE]
}

pub fn matmul_projection_shapes() -> Vec<(usize, usize)> {
    vec![(2048, 2048), (2048, 3072), (2048, 8192), (8192, 2048), (896, 4864), (4864, 896), (3584, 18944), (18944, 3584)]
}

pub fn language_model_head_shape() -> (usize, usize) {
    (2048, VOCABULARY_SIZE)
}
