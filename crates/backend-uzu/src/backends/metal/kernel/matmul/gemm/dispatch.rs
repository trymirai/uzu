use crate::{
    DataType,
    backends::common::{
        Allocation, Backend,
        gpu_types::{
            GemmParams, QuantizationMode,
            gemm::{
                GemmAlignment, GemmComputeKind, GemmInputPrologueKind, GemmOutputTransformKind, GemmTilingConfig,
                GemmWeightPrologueKind,
            },
        },
    },
};

pub enum GemmWeights<'a, B: Backend> {
    FullPrecision {
        weights: &'a Allocation<B>,
    },
    ScaleBias {
        weights: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        biases: &'a Allocation<B>,
        mode: QuantizationMode,
        group_size: u32,
    },
    ScaleZeroPoint {
        weights: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        zero_points: &'a Allocation<B>,
        mode: QuantizationMode,
        group_size: u32,
    },
}

impl<B: Backend> GemmWeights<'_, B> {
    pub fn weight_prologue(&self) -> GemmWeightPrologueKind {
        match self {
            Self::FullPrecision {
                ..
            } => GemmWeightPrologueKind::FullPrecision,
            Self::ScaleBias {
                ..
            } => GemmWeightPrologueKind::ScaleBiasDequant,
            Self::ScaleZeroPoint {
                ..
            } => GemmWeightPrologueKind::ScaleZeroPointDequant,
        }
    }

    pub fn bits_per_weight(&self) -> u32 {
        match self {
            Self::FullPrecision {
                ..
            } => 0,
            Self::ScaleBias {
                mode,
                ..
            }
            | Self::ScaleZeroPoint {
                mode,
                ..
            } => DataType::from(*mode).size_in_bits() as u32,
        }
    }

    pub fn group_size(&self) -> u32 {
        match self {
            Self::FullPrecision {
                ..
            } => 0,
            Self::ScaleBias {
                group_size,
                ..
            }
            | Self::ScaleZeroPoint {
                group_size,
                ..
            } => *group_size,
        }
    }
}

pub struct GemmDispatch<'a, B: Backend> {
    pub tiling_config: GemmTilingConfig,
    pub input_prologue: GemmInputPrologueKind,
    pub compute: GemmComputeKind,
    pub output_transform: GemmOutputTransformKind,
    pub alignment: GemmAlignment,
    pub transpose_weights: bool,
    pub weights: GemmWeights<'a, B>,
    pub weights_offset: usize,
    pub activations: &'a Allocation<B>,
    pub activations_offset: usize,
    pub result: &'a mut Allocation<B>,
    pub params: GemmParams,
    pub group_count_x: u32,
    pub group_count_y: u32,
}

impl<B: Backend> GemmDispatch<'_, B> {
    pub(crate) fn specialization(&self) -> GemmSpecialization {
        GemmSpecialization {
            tiling_config: self.tiling_config,
            input_prologue: self.input_prologue,
            compute: self.compute,
            output_transform: self.output_transform,
            alignment: self.alignment,
            transpose_weights: self.transpose_weights,
            weight_prologue: self.weights.weight_prologue(),
            bits_per_weight: self.weights.bits_per_weight(),
            group_size: self.weights.group_size(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GemmSpecialization {
    pub(crate) tiling_config: GemmTilingConfig,
    pub(crate) input_prologue: GemmInputPrologueKind,
    pub(crate) compute: GemmComputeKind,
    pub(crate) output_transform: GemmOutputTransformKind,
    pub(crate) alignment: GemmAlignment,
    pub(crate) transpose_weights: bool,
    pub(crate) weight_prologue: GemmWeightPrologueKind,
    pub(crate) bits_per_weight: u32,
    pub(crate) group_size: u32,
}

impl GemmSpecialization {
    pub(crate) fn try_validate(self) -> Result<Self, GemmSpecializationError> {
        if self.group_size != 0 && self.tiling_config.threadgroup_k > self.group_size {
            return Err(GemmSpecializationError::ThreadgroupKExceedsGroupSize {
                threadgroup_k: self.tiling_config.threadgroup_k,
                group_size: self.group_size,
            });
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GemmSpecializationError {
    ThreadgroupKExceedsGroupSize {
        threadgroup_k: u32,
        group_size: u32,
    },
}
