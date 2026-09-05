use super::{MatmulA, MatmulArguments, MatmulBKind};
use crate::backends::common::{
    Backend, BufferArg,
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

#[derive(Clone, Copy)]
pub struct MatmulShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub b_transpose: bool,
    pub b_leading_dimension: Option<u32>,
    pub b_kind: MatmulBKind,
    pub b_prologue: GemmBPrologueKind,
    pub b_bits: Option<u32>,
    pub b_group_size: Option<u32>,
    pub signed_codes: bool,
    pub a_full_precision: bool,
    pub gathered: bool,
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
            b_kind: arguments.b.kind(),
            b_prologue: arguments.b.b_prologue(),
            b_bits: arguments.b.bits_per_b(),
            b_group_size: arguments.b.group_size(),
            signed_codes: arguments.b.signed_codes(),
            a_full_precision: matches!(arguments.a, MatmulA::FullPrecision { .. }),
            gathered: arguments.gather_indices.is_some(),
            d_transform: arguments.d_transform.mask(),
        }
    }

    pub fn is_quant(&self) -> bool {
        self.b_prologue != GemmBPrologueKind::FullPrecision
    }
}
