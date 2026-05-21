use super::{specialization::GemmSpecialization, weights::GemmWeights};
use crate::backends::common::{
    Allocation, Backend,
    gpu_types::{
        GemmParams,
        gemm::{GemmAlignment, GemmDTransform, GemmInputPrologueKind, GemmTiling},
    },
};

pub struct GemmDispatch<'a, B: Backend> {
    pub tiling: GemmTiling,
    pub input_prologue: GemmInputPrologueKind,
    pub use_mxu: bool,
    pub output_transform: GemmDTransform,
    pub alignment: GemmAlignment,
    pub transpose_b: bool,
    pub a: &'a Allocation<B>,
    pub a_offset: usize,
    pub b: GemmWeights<'a, B>,
    pub b_offset: usize,
    pub d: &'a mut Allocation<B>,
    pub output_bias: Option<&'a Allocation<B>>,
    pub params: GemmParams,
    pub group_count_x: u32,
    pub group_count_y: u32,
}

impl<B: Backend> GemmDispatch<'_, B> {
    pub(crate) fn specialization(&self) -> GemmSpecialization {
        GemmSpecialization {
            tiling: self.tiling,
            input_prologue: self.input_prologue,
            use_mxu: self.use_mxu,
            output_transform: self.output_transform,
            alignment: self.alignment,
            transpose_b: self.transpose_b,
            weight_prologue: self.b.weight_prologue(),
            bits_per_weight: self.b.bits_per_weight(),
            group_size: self.b.group_size(),
        }
    }
}
