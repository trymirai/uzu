use metal::Buffer as MTLBuffer;

use crate::backends::metal::kernel::mlp::MlpActivationType;

#[derive(Debug)]
pub struct MlpFusedArguments<'a> {
    pub input: &'a MTLBuffer,
    pub input_offset: u64,
    pub weights: &'a MTLBuffer,
    pub output: &'a MTLBuffer,
    pub batch: i32,
    pub input_dim: i32,
    pub hidden_dim: i32,
    pub lda: i32,
    pub ldb: i32,
    pub ldd: i32,
    pub batch_count: i32,
    pub activation: MlpActivationType,
}
