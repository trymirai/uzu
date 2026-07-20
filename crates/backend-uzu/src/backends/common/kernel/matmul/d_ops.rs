use crate::backends::common::{Allocation, Backend, gpu_types::gemm::GemmDTransform};

pub struct MatmulDOps<'a, B: Backend> {
    pub ab_scale: f32,
    pub accumulate: bool,
    pub bias: Option<&'a Allocation<B>>,
    pub rht_factors: Option<&'a Allocation<B>>,
    pub soft_cap: Option<f32>,
}

impl<'a, B: Backend> MatmulDOps<'a, B> {
    pub fn none() -> Self {
        Self {
            ab_scale: 1.0,
            accumulate: false,
            bias: None,
            rht_factors: None,
            soft_cap: None,
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
        }
    }
}
