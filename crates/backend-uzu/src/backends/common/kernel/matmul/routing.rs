use std::num::NonZeroU32;

use super::{MatmulA, MatmulArguments};
use crate::backends::common::{
    Allocation, Backend, BufferArg,
    gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFormat {
    Bf16,
    Int8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct A8ActivationPlan {
    pub activation_group_size: u32,
    pub sum_group_size: Option<u32>,
}

/// The physical A-row layout used by direct expert routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertInput {
    /// A contains one row per token. Route `r` reads row `r / routes_per_token`.
    Tokens,
    /// A already contains one row per route. Route `r` reads row `r`.
    Routes,
}

/// Direct route-major expert selection for a banked B operand.
///
/// `expert_ids` contains at least `M` native `i32` values in token-major, route-major
/// order. Each valid ID selects one matrix from a contiguous B bank. Output row `r`
/// always corresponds to route `r`; there are no public offsets or permutations.
/// Invalid IDs produce an all-zero output row. Empty experts need no representation.
pub struct ExpertRoutes<'a, B: Backend> {
    pub expert_ids: &'a Allocation<B>,
    pub routes_per_token: NonZeroU32,
    pub expert_count: NonZeroU32,
    pub input: ExpertInput,
}

impl<B: Backend> Copy for ExpertRoutes<'_, B> {}

impl<B: Backend> Clone for ExpertRoutes<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Mutually exclusive row and matrix selection modes for matmul.
pub enum MatmulRouting<'a, B: Backend> {
    /// Ordinary dense matmul.
    Dense,
    /// Select one B row for every output element.
    SparseReadout {
        b_rows: &'a Allocation<B>,
    },
    /// Select one B matrix for every output row.
    Experts(ExpertRoutes<'a, B>),
}

impl<B: Backend> Copy for MatmulRouting<'_, B> {}

impl<B: Backend> Clone for MatmulRouting<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, B: Backend> MatmulRouting<'a, B> {
    pub fn sparse_readout_rows(&self) -> Option<&'a Allocation<B>> {
        match self {
            Self::SparseReadout {
                b_rows,
            } => Some(*b_rows),
            Self::Dense | Self::Experts(_) => None,
        }
    }

    pub fn expert_routes(&self) -> Option<ExpertRoutes<'a, B>> {
        match self {
            Self::Experts(routes) => Some(*routes),
            Self::Dense
            | Self::SparseReadout {
                ..
            } => None,
        }
    }
}

#[derive(Clone, Copy)]
pub struct MatmulShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub b_transpose: bool,
    pub b_leading_dimension: Option<u32>,
    pub b_prologue: GemmBPrologueKind,
    pub b_bits: Option<u32>,
    pub b_group_size: Option<u32>,
    pub b_microfloat: Option<crate::backends::common::microfloat::MicrofloatMetadata>,
    pub signed_codes: bool,
    pub a_full_precision: bool,
    pub sparse_readout: bool,
    pub expert_routed: bool,
    pub expert_bias: bool,
    pub d_transform: GemmDTransform,
}

impl MatmulShape {
    pub fn from_arguments<'a, 'b, 'd, B: Backend, TB: BufferArg<'b, B>>(
        arguments: &MatmulArguments<'a, 'b, 'd, B, TB>
    ) -> Self {
        Self {
            m: arguments.m,
            n: arguments.n,
            k: arguments.k,
            b_transpose: arguments.b_transpose,
            b_leading_dimension: arguments.b_leading_dimension,
            b_prologue: arguments.b.b_prologue(),
            b_bits: arguments.b.bits_per_b(),
            b_group_size: arguments.b.group_size(),
            b_microfloat: arguments.b.microfloat_metadata(),
            signed_codes: arguments.b.signed_codes(),
            a_full_precision: matches!(arguments.a, MatmulA::FullPrecision { .. }),
            sparse_readout: arguments.routing.sparse_readout_rows().is_some(),
            expert_routed: arguments.routing.expert_routes().is_some(),
            expert_bias: arguments.d_transform.per_matrix_bias.is_some(),
            d_transform: arguments.d_transform.mask(),
        }
    }

    pub fn is_integer_quantized(&self) -> bool {
        self.b_prologue != GemmBPrologueKind::FullPrecision
    }
}
