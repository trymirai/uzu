use metal::Buffer as MTLBuffer;

#[derive(Debug, Clone)]
pub struct MatmulArguments<'a> {
    pub a: &'a MTLBuffer,
    pub b: &'a MTLBuffer,
    pub c: Option<&'a MTLBuffer>,
    pub d: &'a MTLBuffer,
    /// M dimension - batch/number of tokens (rows of A, rows of D)
    pub batch: i32,
    /// K dimension - input_dim/reduction dimension (cols of A, rows of B)
    pub input_dim: i32,
    /// N dimension - output_dim (cols of B, cols of D)
    pub output_dim: i32,
    pub lda: i32,
    pub ldb: i32,
    pub ldd: i32,
    /// Number of batched matrix multiplications (z-dimension)
    pub batch_count: i32,
    /// Scaling factors for fused addmm (alpha * A @ B + beta * C)
    pub alpha: f32,
    pub beta: f32,
}
