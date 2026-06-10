use crate::backends::common::gpu_types::gemm::{GemmBPrologueKind, GemmDTransform};

/// The buffer-free scalar description of a matmul: everything pipeline selection
/// reads. `encode` derives it from [`MatmulArguments`](super::MatmulArguments) via
/// `task()`; `precompile` builds the same value from load-time state and sweeps
/// `m` with [`with_m`](Self::with_m).
#[derive(Debug, Clone, Copy)]
pub struct MatmulTask {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub b_transpose: bool,
    pub b_offset: usize,
    pub b_leading_dimension: Option<u32>,
    pub b_prologue: GemmBPrologueKind,
    pub bits: Option<u32>,
    pub group_size: Option<u32>,
    pub d_transform: GemmDTransform,
}

impl MatmulTask {
    pub fn with_m(
        self,
        m: u32,
    ) -> Self {
        Self {
            m,
            ..self
        }
    }
}
