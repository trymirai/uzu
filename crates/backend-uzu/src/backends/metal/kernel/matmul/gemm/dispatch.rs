use crate::{
    DataType,
    backends::common::{
        Allocation, Backend,
        gpu_types::{
            GemmParams, QuantizationMode,
            gemm::{
                GemmAlignment, GemmInputPrologueKind, GemmOutputTransformKind, GemmTilingConfig, GemmWeightPrologueKind,
            },
        },
    },
};

pub struct GemmAlignmentAxes {
    pub m: bool,
    pub n: bool,
    pub k: bool,
}

impl GemmAlignment {
    pub fn from_axes(axes: GemmAlignmentAxes) -> Self {
        let mut bits = Self::empty();
        bits.set(Self::M, axes.m);
        bits.set(Self::N, axes.n);
        bits.set(Self::K, axes.k);
        bits
    }
}

#[allow(dead_code)]
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
    pub use_mxu: bool,
    pub output_transform: GemmOutputTransformKind,
    pub alignment: GemmAlignment,
    pub transpose_b: bool,
    pub a: &'a Allocation<B>,
    pub a_offset: usize,
    pub b: GemmWeights<'a, B>,
    pub b_offset: usize,
    pub d: &'a mut Allocation<B>,
    pub params: GemmParams,
    pub group_count_x: u32,
    pub group_count_y: u32,
}

impl<B: Backend> GemmDispatch<'_, B> {
    pub(crate) fn specialization(&self) -> GemmSpecialization {
        GemmSpecialization {
            tiling_config: self.tiling_config,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GemmSpecialization {
    pub(crate) tiling_config: GemmTilingConfig,
    pub(crate) input_prologue: GemmInputPrologueKind,
    pub(crate) use_mxu: bool,
    pub(crate) output_transform: GemmOutputTransformKind,
    pub(crate) alignment: GemmAlignment,
    pub(crate) transpose_b: bool,
    pub(crate) weight_prologue: GemmWeightPrologueKind,
    pub(crate) bits_per_weight: u32,
    pub(crate) group_size: u32,
}

impl GemmSpecialization {
    pub(crate) fn validate(&self) -> Result<(), GemmSpecializationError> {
        if self.group_size != 0 && self.tiling_config.threadgroup_k > self.group_size {
            return Err(GemmSpecializationError::ThreadgroupKExceedsGroupSize {
                threadgroup_k: self.tiling_config.threadgroup_k,
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
        for &(threadgroup_m, threadgroup_n, threadgroup_k) in simdgroup_tile_set(data_type) {
            for align_mn in [true, false] {
                let alignment = GemmAlignment::from_axes(GemmAlignmentAxes {
                    m: align_mn,
                    n: align_mn,
                    k: true,
                });
                out.push(Self {
                    tiling_config: GemmTilingConfig {
                        threadgroup_m,
                        threadgroup_n,
                        threadgroup_k,
                        simdgroups_m: 2,
                        simdgroups_n: 2,
                    },
                    input_prologue: GemmInputPrologueKind::FullPrecision,
                    use_mxu: false,
                    output_transform: GemmOutputTransformKind::Store,
                    alignment,
                    transpose_b: true,
                    weight_prologue: GemmWeightPrologueKind::FullPrecision,
                    bits_per_weight: 0,
                    group_size: 0,
                });
            }
        }

        for &(threadgroup_m, threadgroup_n, simdgroups_m, simdgroups_n) in mxu_tile_set(data_type) {
            for align_m in [true, false] {
                for align_n in [true, false] {
                    for align_k in [true, false] {
                        let alignment = GemmAlignment::from_axes(GemmAlignmentAxes {
                            m: align_m,
                            n: align_n,
                            k: align_k,
                        });
                        for output_transform in [
                            GemmOutputTransformKind::Store,
                            GemmOutputTransformKind::Scale,
                            GemmOutputTransformKind::Accumulate,
                            GemmOutputTransformKind::ScaleAccumulate,
                        ] {
                            out.push(Self {
                                tiling_config: GemmTilingConfig {
                                    threadgroup_m,
                                    threadgroup_n,
                                    threadgroup_k: 32,
                                    simdgroups_m,
                                    simdgroups_n,
                                },
                                input_prologue: GemmInputPrologueKind::FullPrecision,
                                use_mxu: true,
                                output_transform,
                                alignment,
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
        for &(threadgroup_m, threadgroup_n, threadgroup_k, simdgroups_m, simdgroups_n) in
            quant_tile_set(data_type)
        {
            for &group_size in &[32u32, 64, 128] {
                if threadgroup_k > group_size {
                    continue;
                }
                for &bits in &[4u32, 8] {
                    for weight_prologue in [
                        GemmWeightPrologueKind::ScaleBiasDequant,
                        GemmWeightPrologueKind::ScaleZeroPointDequant,
                    ] {
                        for align_n in [true, false] {
                            let alignment = GemmAlignment::from_axes(GemmAlignmentAxes {
                                m: true,
                                n: align_n,
                                k: true,
                            });
                            out.push(Self {
                                tiling_config: GemmTilingConfig {
                                    threadgroup_m,
                                    threadgroup_n,
                                    threadgroup_k,
                                    simdgroups_m,
                                    simdgroups_n,
                                },
                                input_prologue: GemmInputPrologueKind::FullPrecision,
                                use_mxu: false,
                                output_transform: GemmOutputTransformKind::Store,
                                alignment,
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
        out
    }
}

fn simdgroup_tile_set(data_type: DataType) -> &'static [(u32, u32, u32)] {
    match data_type {
        DataType::BF16 => &[(64, 32, 32), (64, 64, 16)],
        DataType::F16 => &[(64, 64, 16), (64, 32, 32)],
        DataType::F32 => &[(32, 64, 16)],
        _ => &[],
    }
}

fn mxu_tile_set(data_type: DataType) -> &'static [(u32, u32, u32, u32)] {
    if !matches!(data_type, DataType::F16 | DataType::BF16) {
        return &[];
    }
    &[(64, 64, 2, 2), (32, 64, 2, 2), (64, 32, 4, 1), (128, 128, 4, 4)]
}

// Quant tiles supported by the unified `Gemm` kernel template. Constrained to
// (BM, BN, BK) shapes that already appear in the FP VARIANTS list — adding
// BM=8 or BK=64 would expand the FP variant grid too.
fn quant_tile_set(data_type: DataType) -> &'static [(u32, u32, u32, u32, u32)] {
    match data_type {
        // (threadgroup_m, threadgroup_n, threadgroup_k, simdgroups_m, simdgroups_n)
        DataType::BF16 => &[(32, 32, 32, 2, 2), (64, 64, 32, 2, 2)],
        DataType::F16 | DataType::F32 => &[(32, 32, 32, 2, 2)],
        _ => &[],
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmSpecializationError {
    ThreadgroupKExceedsGroupSize {
        threadgroup_k: u32,
        group_size: u32,
    },
    QuantizedRequiresSimdgroup,
    QuantizedRequiresTransposedB,
    /// A quantized specialization was routed through the FP `GemmKernel`; should
    /// go through `GemmQuantKernel` instead.
    QuantizedRoutedThroughFpKernel,
}
