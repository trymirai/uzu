use super::error::GemmSpecializationError;
use crate::{
    backends::common::{
        gpu_types::{
            QuantizationMethod,
            gemm::{GemmAlignment, GemmDTransform, GemmTiling, GemmWeightPrologueKind},
        },
        kernel::matmul::MatmulQuantCombo,
    },
    data_type::DataType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GemmSpecialization {
    pub(crate) weights_data_type: DataType,
    pub(crate) tiling: GemmTiling,
    pub(crate) use_mxu: bool,
    pub(crate) output_transform: GemmDTransform,
    pub(crate) alignment: GemmAlignment,
    pub(crate) transpose_b: bool,
    pub(crate) weight_prologue: GemmWeightPrologueKind,
    pub(crate) bits_per_weight: Option<u32>,
    pub(crate) group_size: Option<u32>,
}

impl GemmSpecialization {
    pub(crate) fn validate(&self) -> Result<(), GemmSpecializationError> {
        if let Some(group_size) = self.group_size {
            if self.tiling.block_k() > group_size {
                return Err(GemmSpecializationError::ThreadgroupKExceedsGroupSize {
                    threadgroup_k: self.tiling.block_k(),
                    group_size,
                });
            }
        }
        if self.weight_prologue != GemmWeightPrologueKind::FullPrecision {
            if self.use_mxu {
                return Err(GemmSpecializationError::QuantizedRequiresSimdgroup);
            }
            if !self.transpose_b {
                return Err(GemmSpecializationError::QuantizedRequiresTransposedB);
            }
        }
        Ok(())
    }

    pub(crate) fn precompile_configs(weights_data_type: DataType) -> Vec<Self> {
        let mut out = Vec::new();
        for &tiling in simdgroup_tiling_set(weights_data_type) {
            for align_mn in [true, false] {
                for output_transform in [GemmDTransform::empty(), GemmDTransform::BIAS] {
                    out.push(Self {
                        weights_data_type,
                        tiling,
                        use_mxu: false,
                        output_transform,
                        alignment: GemmAlignment::new(align_mn, align_mn, true),
                        transpose_b: true,
                        weight_prologue: GemmWeightPrologueKind::FullPrecision,
                        bits_per_weight: None,
                        group_size: None,
                    });
                }
            }
        }

        for &tiling in mxu_tiling_set(weights_data_type) {
            for align_m in [true, false] {
                for align_n in [true, false] {
                    for align_k in [true, false] {
                        for output_transform in [
                            GemmDTransform::empty(),
                            GemmDTransform::SCALE,
                            GemmDTransform::ACCUMULATE,
                            GemmDTransform::SCALE | GemmDTransform::ACCUMULATE,
                        ] {
                            out.push(Self {
                                weights_data_type,
                                tiling,
                                use_mxu: true,
                                output_transform,
                                alignment: GemmAlignment::new(align_m, align_n, align_k),
                                transpose_b: true,
                                weight_prologue: GemmWeightPrologueKind::FullPrecision,
                                bits_per_weight: None,
                                group_size: None,
                            });
                        }
                    }
                }
            }
        }
        out
    }

    pub(crate) fn quant_combo_specs(
        weights_data_type: DataType,
        combo: MatmulQuantCombo,
    ) -> Vec<Self> {
        let bits = DataType::from(combo.mode).size_in_bits() as u32;
        let group_size = combo.group_size;
        let weight_prologue = match combo.method {
            QuantizationMethod::ScaleBias => GemmWeightPrologueKind::ScaleBiasDequant,
            QuantizationMethod::ScaleZeroPoint => GemmWeightPrologueKind::ScaleZeroPointDequant,
            QuantizationMethod::ScaleSymmetric => GemmWeightPrologueKind::ScaleSymmetricDequant,
        };
        let mut out = Vec::new();
        for &tiling in quant_tiling_set(weights_data_type) {
            if tiling.block_k() > group_size {
                continue;
            }
            for align_n in [true, false] {
                for output_transform in [GemmDTransform::empty(), GemmDTransform::BIAS] {
                    out.push(Self {
                        weights_data_type,
                        tiling,
                        use_mxu: false,
                        output_transform,
                        alignment: GemmAlignment::new(true, align_n, true),
                        transpose_b: true,
                        weight_prologue,
                        bits_per_weight: Some(bits),
                        group_size: Some(group_size),
                    });
                }
            }
        }
        out
    }
}

fn simdgroup_tiling_set(data_type: DataType) -> &'static [GemmTiling] {
    match data_type {
        DataType::BF16 | DataType::F32 => &[GemmTiling::T64x32x32_2x2, GemmTiling::T64x64x16_2x2],
        _ => &[],
    }
}

fn mxu_tiling_set(data_type: DataType) -> &'static [GemmTiling] {
    if !matches!(data_type, DataType::BF16 | DataType::F32) {
        return &[];
    }
    &[GemmTiling::T64x64x32_2x2, GemmTiling::T32x64x32_2x2, GemmTiling::T64x32x32_4x1, GemmTiling::T128x128x32_4x4]
}

pub(crate) fn quant_tiling_set(data_type: DataType) -> &'static [GemmTiling] {
    match data_type {
        DataType::BF16 => &[
            GemmTiling::T64x64x16_2x2,
            GemmTiling::T8x32x32_1x1,
            GemmTiling::T32x32x32_2x2,
            GemmTiling::T64x32x32_2x2,
            GemmTiling::T64x64x32_2x2,
        ],
        _ => &[],
    }
}
