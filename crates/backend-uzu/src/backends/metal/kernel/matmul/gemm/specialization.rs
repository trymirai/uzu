use super::error::GemmSpecializationError;
use crate::{
    DataType,
    backends::common::gpu_types::gemm::{GemmAlignment, GemmDTransform, GemmTiling, GemmWeightPrologueKind},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GemmSpecialization {
    pub(crate) data_type: DataType,
    pub(crate) tiling: GemmTiling,
    pub(crate) use_mxu: bool,
    pub(crate) output_transform: GemmDTransform,
    pub(crate) alignment: GemmAlignment,
    pub(crate) transpose_b: bool,
    pub(crate) weight_prologue: GemmWeightPrologueKind,
    pub(crate) bits_per_weight: u32,
    pub(crate) group_size: u32,
}

impl GemmSpecialization {
    pub(crate) fn validate(&self) -> Result<(), GemmSpecializationError> {
        if self.group_size != 0 && self.tiling.block_k() > self.group_size {
            return Err(GemmSpecializationError::ThreadgroupKExceedsGroupSize {
                threadgroup_k: self.tiling.block_k(),
                group_size: self.group_size,
            });
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

    pub(crate) fn precompile_configs(data_type: DataType) -> Vec<Self> {
        let mut out = Vec::new();
        for &tiling in simdgroup_tiling_set(data_type) {
            for align_mn in [true, false] {
                for output_transform in [GemmDTransform::empty(), GemmDTransform::BIAS] {
                    out.push(Self {
                        data_type,
                        tiling,
                        use_mxu: false,
                        output_transform,
                        alignment: GemmAlignment::new(align_mn, align_mn, true),
                        transpose_b: true,
                        weight_prologue: GemmWeightPrologueKind::FullPrecision,
                        bits_per_weight: 0,
                        group_size: 0,
                    });
                }
            }
        }

        for &tiling in mxu_tiling_set(data_type) {
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
                                data_type,
                                tiling,
                                use_mxu: true,
                                output_transform,
                                alignment: GemmAlignment::new(align_m, align_n, align_k),
                                transpose_b: true,
                                weight_prologue: GemmWeightPrologueKind::FullPrecision,
                                bits_per_weight: 0,
                                group_size: 0,
                            });
                        }
                    }
                }
            }
        }
        for &tiling in quant_tiling_set(data_type) {
            for &group_size in &[32u32, 64, 128] {
                if tiling.block_k() > group_size {
                    continue;
                }
                for &bits in &[4u32, 8] {
                    for weight_prologue in
                        [GemmWeightPrologueKind::ScaleBiasDequant, GemmWeightPrologueKind::ScaleZeroPointDequant]
                    {
                        for align_n in [true, false] {
                            for output_transform in [GemmDTransform::empty(), GemmDTransform::BIAS] {
                                out.push(Self {
                                    data_type,
                                    tiling,
                                    use_mxu: false,
                                    output_transform,
                                    alignment: GemmAlignment::new(true, align_n, true),
                                    transpose_b: true,
                                    weight_prologue,
                                    bits_per_weight: bits,
                                    group_size,
                                });
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

fn simdgroup_tiling_set(data_type: DataType) -> &'static [GemmTiling] {
    match data_type {
        DataType::BF16 => &[GemmTiling::T64x32x32_2x2, GemmTiling::T64x64x16_2x2],
        DataType::F16 => &[GemmTiling::T64x64x16_2x2, GemmTiling::T64x32x32_2x2],
        _ => &[],
    }
}

fn mxu_tiling_set(data_type: DataType) -> &'static [GemmTiling] {
    if !matches!(data_type, DataType::F16 | DataType::BF16) {
        return &[];
    }
    &[GemmTiling::T64x64x32_2x2, GemmTiling::T32x64x32_2x2, GemmTiling::T64x32x32_4x1, GemmTiling::T128x128x32_4x4]
}

pub(crate) fn quant_tiling_set(data_type: DataType) -> &'static [GemmTiling] {
    match data_type {
        DataType::BF16 => {
            &[GemmTiling::T8x32x32_1x1, GemmTiling::T32x32x32_2x2, GemmTiling::T64x64x32_2x2, GemmTiling::T64x64x64_2x2]
        },
        DataType::F16 => &[GemmTiling::T8x32x32_1x1, GemmTiling::T32x32x32_2x2],
        _ => &[],
    }
}
