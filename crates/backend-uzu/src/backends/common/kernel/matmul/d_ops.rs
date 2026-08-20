use crate::backends::common::{Allocation, Backend, gpu_types::gemm::GemmDTransform};

/// Fused gated-activation epilogue parameters for [GemmDTransform::GATE_ACT_MUL].
///
/// Only SiLU is supported; `activation_alpha` overrides the default alpha.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateActMulDOps {
    pub activation_alpha: Option<f32>,
    pub gate_clipping: Option<(f32, f32)>,
    pub value_clipping: Option<(f32, f32)>,
}

pub struct MatmulDOps<'a, B: Backend> {
    pub ab_scale: f32,
    pub accumulate: bool,
    pub bias: Option<&'a Allocation<B>>,
    /// Optional `[matrix_count, N]` bias bank selected by expert routing.
    pub per_matrix_bias: Option<&'a Allocation<B>>,
    pub rht_factors: Option<&'a Allocation<B>>,
    pub soft_cap: Option<f32>,
    pub gate_act: Option<GateActMulDOps>,
}

impl<'a, B: Backend> MatmulDOps<'a, B> {
    pub fn none() -> Self {
        Self {
            ab_scale: 1.0,
            accumulate: false,
            bias: None,
            per_matrix_bias: None,
            rht_factors: None,
            soft_cap: None,
            gate_act: None,
        }
    }

    pub fn mask(&self) -> GemmDTransform {
        let mut m = GemmDTransform::empty();
        if self.ab_scale != 1.0 {
            m |= GemmDTransform::SCALE;
        }
        if self.accumulate {
            m |= GemmDTransform::ACCUMULATE;
        }
        if self.bias.is_some() {
            m |= GemmDTransform::BIAS;
        }
        if self.rht_factors.is_some() {
            m |= GemmDTransform::RHT;
        }
        if self.soft_cap.is_some() {
            m |= GemmDTransform::SOFT_CAP;
        }
        if self.gate_act.is_some() {
            m |= GemmDTransform::GATE_ACT_MUL;
        }
        m
    }

    pub fn without(
        self,
        bits: GemmDTransform,
    ) -> Self {
        Self {
            ab_scale: if bits.contains(GemmDTransform::SCALE) {
                1.0
            } else {
                self.ab_scale
            },
            accumulate: if bits.contains(GemmDTransform::ACCUMULATE) {
                false
            } else {
                self.accumulate
            },
            bias: if bits.contains(GemmDTransform::BIAS) {
                None
            } else {
                self.bias
            },
            per_matrix_bias: if bits.contains(GemmDTransform::BIAS) {
                None
            } else {
                self.per_matrix_bias
            },
            rht_factors: if bits.contains(GemmDTransform::RHT) {
                None
            } else {
                self.rht_factors
            },
            soft_cap: if bits.contains(GemmDTransform::SOFT_CAP) {
                None
            } else {
                self.soft_cap
            },
            gate_act: if bits.contains(GemmDTransform::GATE_ACT_MUL) {
                None
            } else {
                self.gate_act
            },
        }
    }
}
