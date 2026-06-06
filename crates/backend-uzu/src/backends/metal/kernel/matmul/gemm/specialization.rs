use super::{
    error::GemmSpecializationError,
    kernel::{select_mxu_quant_tiling, select_mxu_tiling, select_quant_tiling, select_simdgroup_tiling},
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

    pub(crate) fn full_precision_combo_specs(
        weights_data_type: DataType,
        n: u32,
        k: u32,
        output_transform: GemmDTransform,
        use_mxu: bool,
    ) -> Vec<Self> {
        let tiling_supported = if use_mxu {
            !mxu_tiling_set(weights_data_type).is_empty()
        } else {
            !simdgroup_tiling_set(weights_data_type).is_empty()
        };
        if !tiling_supported {
            return Vec::new();
        }

        let mut tilings: Vec<GemmTiling> = Vec::new();
        for &m in &QUANT_PREHEAT_PROBE_MS {
            let tiling = if use_mxu {
                select_mxu_tiling(m, n)
            } else {
                select_simdgroup_tiling(m, n, k)
            };
            if !tilings.contains(&tiling) {
                tilings.push(tiling);
            }
        }

        let mut transforms = vec![GemmDTransform::empty()];
        if !output_transform.is_empty() {
            transforms.push(output_transform);
        }

        let mut out = Vec::new();
        for tiling in tilings {
            let align_n = n.is_multiple_of(tiling.block_n());
            let align_k = k.is_multiple_of(tiling.block_k());
            for &output_transform in &transforms {
                for align_m in [true, false] {
                    let spec = Self {
                        weights_data_type,
                        tiling,
                        use_mxu,
                        output_transform,
                        alignment: GemmAlignment::new(align_m, align_n, align_k),
                        transpose_b: true,
                        b_prologue: GemmBPrologueKind::FullPrecision,
                        bits_per_b: None,
                        group_size: None,
                    };
                    if spec.validate().is_ok() {
                        out.push(spec);
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
        assert!(specs.len() <= 20, "expected a tight preheat set, got {}", specs.len());
        for spec in &specs {
            assert!(spec.use_mxu);
            assert!(spec.tiling.is_mxu_variant());
            assert!(spec.validate().is_ok());
        }
    }

    #[test]
    fn full_precision_specs_are_shape_specific_and_tight() {
        let plain =
            GemmSpecialization::full_precision_combo_specs(DataType::BF16, 3584, 1024, GemmDTransform::empty(), false);
        assert!(!plain.is_empty());
        assert!(plain.len() <= 16, "expected a tight preheat set, got {}", plain.len());
        for spec in &plain {
            assert!(!spec.use_mxu);
            assert_eq!(spec.b_prologue, GemmBPrologueKind::FullPrecision);
            assert_eq!(spec.output_transform, GemmDTransform::empty());
            assert!(spec.validate().is_ok());
        }

        let biased =
            GemmSpecialization::full_precision_combo_specs(DataType::BF16, 3584, 1024, GemmDTransform::BIAS, false);
        let transforms: std::collections::HashSet<_> = biased.iter().map(|spec| spec.output_transform).collect();
        assert!(transforms.contains(&GemmDTransform::empty()));
        assert!(transforms.contains(&GemmDTransform::BIAS));
        assert_eq!(transforms.len(), 2, "only empty + the layer transform should be preheated");
    }

    #[test]
    fn full_precision_mxu_emits_only_mxu_tilings() {
        let specs =
            GemmSpecialization::full_precision_combo_specs(DataType::BF16, 3584, 1024, GemmDTransform::BIAS, true);
        assert!(!specs.is_empty());
        for spec in &specs {
            assert!(spec.use_mxu);
            assert!(spec.tiling.is_mxu_variant());
            assert!(spec.validate().is_ok());
        }
    }
}
