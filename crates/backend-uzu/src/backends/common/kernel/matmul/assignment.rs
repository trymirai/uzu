use std::num::NonZeroU32;

use crate::backends::common::{Allocation, Backend};

/// Indirect row selection from flattened logical `u32` indices.
///
/// Assignment `r` selects `indices[r] / index_divisor`. A divisor of one is an ordinary row map;
/// larger divisors let repeated logical items share one physical source row without another map.
/// The allocation contains at least `M` entries, and resolved indices must name physical rows in the
/// selected operand.
pub struct MatmulRowMap<'a, B: Backend> {
    pub indices: &'a Allocation<B>,
    pub index_divisor: NonZeroU32,
}

impl<B: Backend> Copy for MatmulRowMap<'_, B> {}

impl<B: Backend> Clone for MatmulRowMap<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Selection of one matrix or an assignment-sorted matrix bank.
pub enum MatmulMatrixMap<'a, B: Backend> {
    /// Every assignment uses the sole B matrix.
    Shared,
    /// `offsets[g]..offsets[g + 1]` contains the assignments for matrix `g`.
    ///
    /// The allocation contains `G + 1` native `u32` values. Offsets are nondecreasing, start at zero,
    /// and end at the matmul `M`; repeated offsets encode empty matrices without route-sized matrix
    /// indices.
    Segmented {
        offsets: &'a Allocation<B>,
        matrix_count: NonZeroU32,
    },
}

impl<B: Backend> Copy for MatmulMatrixMap<'_, B> {}

impl<B: Backend> Clone for MatmulMatrixMap<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Compact matrix assignments shared by gathered GEMV and grouped GEMM.
///
/// For assignment `r`, the optional row maps select A and D rows, while `matrices` selects a B matrix.
/// B is a contiguous `[G, N, K]` bank when transposed and `[G, K, N]` otherwise, using the configured
/// leading dimension within each matrix; `G` is one for `Shared`. Matrix scales have shape `[G]` and
/// matrix biases `[G, N]`, both in the weights data type. The epilogue computes
/// `ab_scale * matrix_scale[g] * dot`, then adds accumulated D, shared bias, and matrix bias before
/// soft-capping. Repeated source rows and matrix assignments are valid. Resolved destination rows
/// must be a permutation of `0..M`, because backends write them without atomic reduction.
pub struct GatheredMatmul<'a, B: Backend> {
    pub source_rows: Option<MatmulRowMap<'a, B>>,
    pub matrices: MatmulMatrixMap<'a, B>,
    pub destination_rows: Option<MatmulRowMap<'a, B>>,
    pub matrix_scales: Option<&'a Allocation<B>>,
    pub matrix_biases: Option<&'a Allocation<B>>,
}

impl<B: Backend> Copy for GatheredMatmul<'_, B> {}

impl<B: Backend> Clone for GatheredMatmul<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

/// How logical matmul outputs select their operands and destination rows.
pub enum MatmulRouting<'a, B: Backend> {
    Dense,
    /// Existing sparse readout: every output element selects a B row from an `[M, N]` index matrix.
    SparseReadout {
        b_rows: &'a Allocation<B>,
    },
    /// Assignment-level A-row, B-matrix, and D-row selection with `O(M + G)` metadata.
    Gathered(GatheredMatmul<'a, B>),
}

impl<B: Backend> Copy for MatmulRouting<'_, B> {}

impl<B: Backend> Clone for MatmulRouting<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Routing class used for backend path selection without retaining routing buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulRoutingKind {
    Dense,
    SparseReadout,
    Gathered,
}

impl<B: Backend> MatmulRouting<'_, B> {
    pub fn kind(&self) -> MatmulRoutingKind {
        match self {
            Self::Dense => MatmulRoutingKind::Dense,
            Self::SparseReadout {
                ..
            } => MatmulRoutingKind::SparseReadout,
            Self::Gathered(_) => MatmulRoutingKind::Gathered,
        }
    }
}
