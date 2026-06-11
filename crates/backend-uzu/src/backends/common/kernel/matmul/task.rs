use super::{d_ops::MatmulDOps, matmul_b::MatmulB};
use crate::backends::common::{
    AsBufferRangeRef, Backend,
    gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
};

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
    /// Derives the task from the matmul shape plus its `b`/`d` operands. Both
    /// [`MatmulArguments::task`](super::MatmulArguments::task) and every
    /// `precompile` site funnel through here, so a preheated pipeline always
    /// matches the one a later `encode` selects.
    #[allow(clippy::too_many_arguments)]
    pub fn new<B: Backend, TB: AsBufferRangeRef>(
        m: u32,
        n: u32,
        k: u32,
        b_transpose: bool,
        b_offset: usize,
        b_leading_dimension: Option<u32>,
        b: &MatmulB<'_, B, TB>,
        d_transform: &MatmulDOps<'_, B>,
    ) -> Self {
        Self {
            m,
            n,
            k,
            b_transpose,
            b_offset,
            b_leading_dimension,
            b_prologue: b.b_prologue(),
            bits: b.bits_per_b(),
            group_size: b.group_size(),
            d_transform: d_transform.mask(),
        }
    }

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
