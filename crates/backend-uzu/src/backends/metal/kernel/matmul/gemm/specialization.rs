use super::error::GemmSpecializationError;
use crate::{
    backends::common::{
        gpu_types::{
            QuantizationMethod,
            gemm::{GemmAlignment, GemmBPrologueKind, GemmDTransform, GemmTiling},
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
    pub(crate) b_prologue: GemmBPrologueKind,
    pub(crate) bits_per_b: Option<u32>,
    pub(crate) group_size: Option<u32>,
}

impl GemmSpecialization {
    pub(crate) fn validate(&self) -> Result<(), GemmSpecializationError> {
        if self.use_mxu != self.tiling.is_mxu_variant() {
            return Err(GemmSpecializationError::TilingUseMxuMismatch {
                tiling: self.tiling,
                use_mxu: self.use_mxu,
            });
        }
        if self.use_mxu
            && self.b_prologue != GemmBPrologueKind::FullPrecision
            && let Some(group_size) = self.group_size
            && !self.tiling.fits_quant_group_size(group_size)
        {
            return Err(GemmSpecializationError::MxuQuantTileTooLarge {
                tiling: self.tiling,
                group_size,
            });
        }
        if !self.use_mxu
            && let Some(group_size) = self.group_size
        {
            let simdgroup_block_k = self.tiling.simdgroup_block_k();
            if simdgroup_block_k > group_size {
                return Err(GemmSpecializationError::SimdgroupKExceedsGroupSize {
                    simdgroup_k: simdgroup_block_k,
                    group_size,
                });
            }
        }
        if self.b_prologue != GemmBPrologueKind::FullPrecision && !self.transpose_b {
            return Err(GemmSpecializationError::QuantizedRequiresTransposedB);
        }
        Ok(())
    }

    pub(crate) fn precompile_configs(weights_data_type: DataType) -> Vec<Self> {
        let mut out = Vec::new();
        for &tiling in simdgroup_tiling_set(weights_data_type) {
            for align_mn in [true, false] {
                for output_transform in [
                    GemmDTransform::empty(),
                    GemmDTransform::BIAS,
                    GemmDTransform::RHT,
                    GemmDTransform::BIAS | GemmDTransform::RHT,
                ] {
                    out.push(Self {
                        weights_data_type,
                        tiling,
                        use_mxu: false,
                        output_transform,
                        alignment: GemmAlignment::new(align_mn, align_mn, true),
                        transpose_b: true,
                        b_prologue: GemmBPrologueKind::FullPrecision,
                        bits_per_b: None,
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
                            GemmDTransform::BIAS,
                            GemmDTransform::BIAS | GemmDTransform::SCALE,
                            GemmDTransform::BIAS | GemmDTransform::ACCUMULATE,
                            GemmDTransform::BIAS | GemmDTransform::SCALE | GemmDTransform::ACCUMULATE,
                            GemmDTransform::RHT,
                            GemmDTransform::RHT | GemmDTransform::SCALE,
                            GemmDTransform::RHT | GemmDTransform::ACCUMULATE,
                            GemmDTransform::RHT | GemmDTransform::SCALE | GemmDTransform::ACCUMULATE,
                            GemmDTransform::BIAS | GemmDTransform::RHT,
                            GemmDTransform::BIAS | GemmDTransform::RHT | GemmDTransform::SCALE,
                            GemmDTransform::BIAS | GemmDTransform::RHT | GemmDTransform::ACCUMULATE,
                            GemmDTransform::BIAS
                                | GemmDTransform::RHT
                                | GemmDTransform::SCALE
                                | GemmDTransform::ACCUMULATE,
                        ] {
                            out.push(Self {
                                weights_data_type,
                                tiling,
                                use_mxu: true,
                                output_transform,
                                alignment: GemmAlignment::new(align_m, align_n, align_k),
                                transpose_b: true,
                                b_prologue: GemmBPrologueKind::FullPrecision,
                                bits_per_b: None,
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
        let b_prologue = match combo.method {
            QuantizationMethod::ScaleBias => GemmBPrologueKind::ScaleBiasDequant,
            QuantizationMethod::ScaleZeroPoint => GemmBPrologueKind::ScaleZeroPointDequant,
            QuantizationMethod::ScaleSymmetric => GemmBPrologueKind::ScaleSymmetricDequant,
        };
        let mut out = Vec::new();
        for &tiling in quant_tiling_set(weights_data_type) {
            if tiling.simdgroup_block_k() > group_size {
                continue;
            }
            for align_n in [true, false] {
                for output_transform in [
                    GemmDTransform::empty(),
                    GemmDTransform::BIAS,
                    GemmDTransform::RHT,
                    GemmDTransform::BIAS | GemmDTransform::RHT,
                ] {
                    out.push(Self {
                        weights_data_type,
                        tiling,
                        use_mxu: false,
                        output_transform,
                        alignment: GemmAlignment::new(true, align_n, true),
                        transpose_b: true,
                        b_prologue,
                        bits_per_b: Some(bits),
                        group_size: Some(group_size),
                    });
                }
            }
        }
        for &tiling in mxu_tiling_set(weights_data_type) {
            if !group_size.is_multiple_of(32) {
                continue;
            }
            if !tiling.fits_quant_group_size(group_size) {
                continue;
            }
            for align_m in [true, false] {
                for align_n in [true, false] {
                    for output_transform in [
                        GemmDTransform::empty(),
                        GemmDTransform::SCALE,
                        GemmDTransform::BIAS,
                        GemmDTransform::RHT,
                        GemmDTransform::BIAS | GemmDTransform::RHT,
                        GemmDTransform::BIAS | GemmDTransform::SCALE,
                    ] {
                        out.push(Self {
                            weights_data_type,
                            tiling,
                            use_mxu: true,
                            output_transform,
                            alignment: GemmAlignment::new(align_m, align_n, true),
                            transpose_b: true,
                            b_prologue,
                            bits_per_b: Some(bits),
                            group_size: Some(group_size),
                        });
                    }
                }
            }
        }
        out
    }
}

fn simdgroup_tiling_set(data_type: DataType) -> &'static [GemmTiling] {
    match data_type {
        DataType::BF16 | DataType::F32 => {
            &[GemmTiling::Tile64x32x32_Simdgroups2x2, GemmTiling::Tile64x64x16_Simdgroups2x2]
        },
        _ => &[],
    }
}

fn mxu_tiling_set(data_type: DataType) -> &'static [GemmTiling] {
    if !matches!(data_type, DataType::BF16 | DataType::F32) {
        return &[];
    }
    &[
        GemmTiling::Tile64x64x256_Simdgroups2x2,
        GemmTiling::Tile32x64x256_Simdgroups2x2,
        GemmTiling::Tile64x32x256_Simdgroups4x1,
        GemmTiling::Tile128x128x256_Simdgroups4x4,
    ]
}

pub(crate) fn quant_tiling_set(data_type: DataType) -> &'static [GemmTiling] {
    match data_type {
        DataType::BF16 => &[
            GemmTiling::Tile8x32x32_Simdgroups1x1,
            GemmTiling::Tile32x32x32_Simdgroups2x2,
            GemmTiling::Tile64x32x32_Simdgroups2x2,
            GemmTiling::Tile64x64x32_Simdgroups2x2,
        ],
        _ => &[],
    }
}
