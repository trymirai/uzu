use super::{
    error::GemmSpecializationError,
    kernel::{select_mxu_quant_tiling, select_quant_tiling},
};
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

const QUANT_PREHEAT_PROBE_MS: [u32; 4] = [4096, 128, 48, 8];

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
        n: u32,
        k: u32,
        use_mxu: bool,
    ) -> Vec<Self> {
        let bits = DataType::from(combo.mode).size_in_bits() as u32;
        let group_size = combo.group_size;
        let b_prologue = match combo.method {
            QuantizationMethod::ScaleBias => GemmBPrologueKind::ScaleBiasDequant,
            QuantizationMethod::ScaleZeroPoint => GemmBPrologueKind::ScaleZeroPointDequant,
            QuantizationMethod::ScaleSymmetric => GemmBPrologueKind::ScaleSymmetricDequant,
        };

        let mut tilings: Vec<GemmTiling> = Vec::new();
        for &m in &QUANT_PREHEAT_PROBE_MS {
            let tiling = if use_mxu {
                select_mxu_quant_tiling(m, n, group_size)
            } else {
                select_quant_tiling(m, n, group_size)
            };
            if !tilings.contains(&tiling) {
                tilings.push(tiling);
            }
        }

        let mut out = Vec::new();
        for tiling in tilings {
            let align_n = n.is_multiple_of(tiling.block_n());
            let align_k = k.is_multiple_of(tiling.block_k());
            for output_transform in [GemmDTransform::empty(), GemmDTransform::BIAS, GemmDTransform::RHT] {
                for align_m in [true, false] {
                    let spec = Self {
                        weights_data_type,
                        tiling,
                        use_mxu,
                        output_transform,
                        alignment: GemmAlignment::new(align_m, align_n, align_k),
                        transpose_b: true,
                        b_prologue,
                        bits_per_b: Some(bits),
                        group_size: Some(group_size),
                    };
                    if spec.validate().is_ok() {
                        out.push(spec);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::common::gpu_types::QuantizationMode;

    fn combo() -> MatmulQuantCombo {
        MatmulQuantCombo {
            method: QuantizationMethod::ScaleZeroPoint,
            mode: QuantizationMode::U4,
            group_size: 32,
        }
    }

    #[test]
    fn quant_specs_are_shape_specific_and_tight() {
        let specs = GemmSpecialization::quant_combo_specs(DataType::BF16, combo(), 3584, 1024, false);
        assert!(!specs.is_empty());
        assert!(specs.len() <= 16, "expected a tight preheat set, got {}", specs.len());
        for spec in &specs {
            assert!(!spec.use_mxu);
            assert_eq!(spec.bits_per_b, Some(4));
            assert_eq!(spec.group_size, Some(32));
            assert_ne!(spec.b_prologue, GemmBPrologueKind::FullPrecision);
            assert!(spec.validate().is_ok());
        }
        let tilings: std::collections::HashSet<_> = specs.iter().map(|s| s.tiling).collect();
        assert!(tilings.contains(&GemmTiling::Tile32x32x32_Simdgroups2x2));
        assert!(tilings.contains(&GemmTiling::Tile8x32x32_Simdgroups1x1));
    }

    #[test]
    fn mxu_path_emits_only_mxu_tilings() {
        let specs = GemmSpecialization::quant_combo_specs(DataType::BF16, combo(), 3584, 1024, true);
        assert!(!specs.is_empty());
        // 3 MXU tilings (m≥256 / 64≤m<256 / m<64) × 3 transforms × 2 align_m.
        assert!(specs.len() <= 20, "expected a tight preheat set, got {}", specs.len());
        for spec in &specs {
            assert!(spec.use_mxu);
            assert!(spec.tiling.is_mxu_variant());
            assert!(spec.validate().is_ok());
        }
    }
}
