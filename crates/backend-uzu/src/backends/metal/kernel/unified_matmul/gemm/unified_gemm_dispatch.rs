use crate::backends::{
    common::{
        Backend, Borrowed,
        gpu_types::unified_gemm::{
            GemmAlignment, GemmComputeKind, GemmInputPrologueKind, GemmOutputTransformKind, GemmTilingConfig,
        },
    },
    metal::Metal,
};

use super::{GemmWeights, UnifiedGemmSpecialization};

type DenseBuffer = <Metal as Backend>::DenseBuffer;

pub(crate) type GemmWeightsBorrowed<'a> = GemmWeights<Borrowed<'a, DenseBuffer>>;

pub(crate) struct UnifiedGemmDispatch<'a> {
    pub(crate) tiling_config: GemmTilingConfig,
    pub(crate) input_prologue: GemmInputPrologueKind,
    pub(crate) compute: GemmComputeKind,
    pub(crate) output_transform: GemmOutputTransformKind,
    pub(crate) alignment: GemmAlignment,
    pub(crate) weights: GemmWeightsBorrowed<'a>,
    pub(crate) activations: &'a DenseBuffer,
    pub(crate) activations_offset: usize,
    pub(crate) result: &'a mut DenseBuffer,
    pub(crate) group_count_x: u32,
    pub(crate) group_count_y: u32,
}

impl UnifiedGemmDispatch<'_> {
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
