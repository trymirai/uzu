pub const MINIMUM_MODEL_DIMENSION: usize = 512;
pub const MAXIMUM_MODEL_DIMENSION: usize = 8192;
pub const MODEL_DIMENSION_STEP: usize = 128;

pub fn model_dimensions() -> impl Iterator<Item = usize> + Clone {
    (MINIMUM_MODEL_DIMENSION..=MAXIMUM_MODEL_DIMENSION).step_by(MODEL_DIMENSION_STEP)
}

pub const DECODE_TOKEN_COUNT: usize = 1;

pub const REAL_TOKEN_COUNTS: [usize; 4] = [DECODE_TOKEN_COUNT, 128, 512, 2048];
