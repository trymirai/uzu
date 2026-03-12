use crate::backends::common::Backend;

#[derive(Debug)]
pub struct MatmulArguments<'a, B: Backend> {
    pub a: &'a B::Buffer,
    /// Byte offset into `a` (used for slicing the batch dimension).
    pub a_offset: u64,
    pub b: &'a B::Buffer,
    pub d: &'a mut B::Buffer,
    pub bias: Option<&'a B::Buffer>,
    /// M dimension - batch/number of tokens (rows of A, rows of D)
    pub batch: i32,
    /// K dimension - input_dim/reduction dimension (cols of A, rows of B)
    pub input_dim: i32,
    /// N dimension - output_dim (cols of B, cols of D)
    pub output_dim: i32,
}
