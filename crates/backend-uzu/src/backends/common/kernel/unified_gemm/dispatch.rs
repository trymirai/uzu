use crate::backends::common::{
    Allocation, Backend,
    gpu_types::{
        GemmParams,
        unified_gemm::{
            GemmAlignment, GemmComputeKind, GemmInputPrologueKind, GemmOutputTransformKind, GemmTilingConfig,
        },
    },
    kernel::unified_gemm::{GemmWeights, UnifiedGemmSpecialization},
};

pub struct UnifiedGemmDispatch<'a, B: Backend> {
    pub tiling_config: GemmTilingConfig,
    pub input_prologue: GemmInputPrologueKind,
    pub compute: GemmComputeKind,
    pub output_transform: GemmOutputTransformKind,
    pub alignment: GemmAlignment,
    pub weights: GemmWeights<'a, B>,
    pub activations: &'a Allocation<B>,
    pub activations_offset: usize,
    pub result: &'a mut Allocation<B>,
    pub params: GemmParams,
    pub group_count_x: u32,
    pub group_count_y: u32,
}

impl<B: Backend> UnifiedGemmDispatch<'_, B> {
    pub(crate) fn specialization(&self) -> UnifiedGemmSpecialization {
        UnifiedGemmSpecialization {
            tiling_config: self.tiling_config,
            input_prologue: self.input_prologue,
            compute: self.compute,
            output_transform: self.output_transform,
            alignment: self.alignment,
            weight_prologue: self.weights.weight_prologue(),
            bits_per_weight: self.weights.bits_per_weight(),
            group_size: self.weights.group_size(),
        }
    }
}
