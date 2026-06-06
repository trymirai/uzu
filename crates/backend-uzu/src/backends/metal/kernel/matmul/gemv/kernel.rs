use std::{
    collections::{HashMap, hash_map::Entry},
    sync::OnceLock,
};

use crate::{
    backends::{
        common::{
            Allocation, AsBufferRangeRef, Buffer, Encoder,
            gpu_types::{
                QuantizationMethod,
                gemm::{GemmBPrologueKind, GemmDTransform},
            },
            kernel::matmul::{MatmulArguments, MatmulB, MatmulError, MatmulQuantCombo},
        },
        metal::{Metal, context::MetalContext, kernel::GemvMetalKernel},
    },
    data_type::DataType,
};

const FP_BLOCK: u32 = 128;
const QMV_RESULTS_PER_SIMDGROUP: u32 = 4;

const DEFAULT_GEMV_MAX_BATCH: u32 = 8;
static GEMV_MAX_BATCH: OnceLock<u32> = OnceLock::new();

fn max_gemv_batch_threshold() -> u32 {
    *GEMV_MAX_BATCH.get_or_init(|| {
        std::env::var("UZU_GEMV_MAX_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_GEMV_MAX_BATCH)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GemvSpecialization {
    b_prologue: GemmBPrologueKind,
    group_size: u32,
    bits: u32,
    output_transform: GemmDTransform,
    input_aligned: bool,
    k_split: u32,
    num_simdgroups: u32,
}

impl GemvSpecialization {
    pub(crate) fn select<TB: AsBufferRangeRef>(
        args: &MatmulArguments<Metal, TB>,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Option<GemvSpecialization> {
        if !args.b_transpose || args.b_offset != 0 {
            return None;
        }
        let is_quant = !matches!(args.b, MatmulB::FullPrecision { .. });
        let bad_leading_dimension = if is_quant {
            args.b_leading_dimension.is_some()
        } else {
            args.b_leading_dimension.is_some_and(|ld| ld != args.k)
        };
        if bad_leading_dimension {
            return None;
        }
        if args.d_transform.accumulate && !args.n.is_multiple_of(32) {
            return None;
        }
        if args.d_transform.rht_factors.is_some() && !args.n.is_multiple_of(32) {
            return None;
        }
        if is_quant {
            if args.n < QMV_RESULTS_PER_SIMDGROUP || args.m >= 5 {
                return None;
            }
        } else {
            let mixed_precision = weights_data_type == DataType::F32
                && (input_data_type != DataType::F32 || output_data_type != DataType::F32);
            if mixed_precision || args.n < QMV_RESULTS_PER_SIMDGROUP || args.m > max_gemv_batch_threshold() {
                return None;
            }
        }

        let bits = args.b.bits_per_b().unwrap_or(0);
        let block_size = if !is_quant {
            FP_BLOCK
        } else if bits == 4 {
            512
        } else {
            256
        };
        let input_aligned = args.k.is_multiple_of(block_size);
        let has_rht = args.d_transform.rht_factors.is_some();
        let num_simdgroups = 8;
        let k_split = if is_quant || has_rht {
            1
        } else {
            fp_k_split(args.n, args.k, input_aligned)
        };
        Some(GemvSpecialization {
            b_prologue: args.b.b_prologue(),
            group_size: args.b.group_size().unwrap_or(0),
            bits,
            output_transform: args.d_transform.mask(),
            input_aligned,
            k_split,
            num_simdgroups,
        })
    }

    fn quant_combo_specs(combo: MatmulQuantCombo) -> Vec<GemvSpecialization> {
        let bits = DataType::from(combo.mode).size_in_bits() as u32;
        let group_size = combo.group_size;
        let b_prologue = match combo.method {
            QuantizationMethod::ScaleBias => GemmBPrologueKind::ScaleBiasDequant,
            QuantizationMethod::ScaleZeroPoint => GemmBPrologueKind::ScaleZeroPointDequant,
            QuantizationMethod::ScaleSymmetric => GemmBPrologueKind::ScaleSymmetricDequant,
            QuantizationMethod::LloydMax => return Vec::new(),
        };
        let mut out = Vec::new();
        for output_transform in [
            GemmDTransform::empty(),
            GemmDTransform::BIAS,
            GemmDTransform::RHT,
            GemmDTransform::BIAS | GemmDTransform::RHT,
        ] {
            for input_aligned in [true, false] {
                out.push(GemvSpecialization {
                    b_prologue,
                    group_size,
                    bits,
                    output_transform,
                    input_aligned,
                    k_split: 1,
                    num_simdgroups: 8,
                });
            }
        }
        out
    }
}

fn rows_per_threadgroup(
    k_split: u32,
    num_simdgroups: u32,
) -> u32 {
    (num_simdgroups / k_split) * QMV_RESULTS_PER_SIMDGROUP
}

fn fp_k_split(
    n: u32,
    k: u32,
    input_aligned: bool,
) -> u32 {
    if !input_aligned || n >= 4096 {
        1
    } else if k >= 16 * n || n <= 512 {
        8
    } else if n <= 1024 || k >= 3072 {
        4
    } else {
        2
    }
}

pub(crate) struct GemvDispatch {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    pipelines: HashMap<GemvSpecialization, GemvMetalKernel>,
}

impl GemvDispatch {
    pub(crate) fn new(
        _context: &MetalContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, MatmulError<Metal>> {
        Ok(Self {
            weights_data_type,
            input_data_type,
            output_data_type,
            pipelines: HashMap::new(),
        })
    }

    pub(crate) fn preheat_quant_combo(
        &mut self,
        context: &MetalContext,
        combo: MatmulQuantCombo,
    ) -> Result<(), MatmulError<Metal>> {
        if self.weights_data_type != DataType::BF16 {
            return Ok(());
        }
        for specialization in GemvSpecialization::quant_combo_specs(combo) {
            self.get_or_create(context, specialization)?;
        }
        Ok(())
    }

    fn get_or_create(
        &mut self,
        context: &MetalContext,
        specialization: GemvSpecialization,
    ) -> Result<&GemvMetalKernel, MatmulError<Metal>> {
        match self.pipelines.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = GemvMetalKernel::new(
                    context,
                    self.input_data_type,
                    self.weights_data_type,
                    self.output_data_type,
                    specialization.b_prologue,
                    specialization.group_size,
                    specialization.bits,
                    specialization.k_split,
                    specialization.input_aligned,
                    specialization.num_simdgroups,
                    specialization.output_transform,
                )
                .map_err(MatmulError::BackendError)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    pub(crate) fn encode<'a, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<'a, Metal, TB>,
        specialization: GemvSpecialization,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let ab_scale = arguments.d_transform.ab_scale;
        let output_bias = arguments.d_transform.bias;
        let rht_factors = arguments.d_transform.rht_factors;

        let MatmulArguments {
            a,
            a_offset,
            b,
            d,
            m,
            n,
            k,
            ..
        } = arguments;

        let group_count_x = n.div_ceil(rows_per_threadgroup(specialization.k_split, specialization.num_simdgroups));

        let context = encoder.context();
        let pipeline = self.get_or_create(context, specialization)?;

        match b {
            MatmulB::FullPrecision {
                b: weights,
            } => {
                pipeline.encode(
                    weights,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    (a, a_offset),
                    &mut *d,
                    output_bias,
                    rht_factors,
                    k,
                    n,
                    m,
                    ab_scale,
                    group_count_x,
                    encoder,
                );
            },
            quant_b @ (MatmulB::ScaleBiasDequant {
                ..
            }
            | MatmulB::ScaleZeroPointDequant {
                ..
            }
            | MatmulB::ScaleSymmetricDequant {
                ..
            }
            | MatmulB::LloydMaxDequant {
                ..
            }) => {
                let (weights, scales, zero_points, biases, codebook, bias_indices, bias_codebook) = match quant_b {
                    MatmulB::ScaleBiasDequant {
                        b: w,
                        scales,
                        biases,
                        ..
                    } => (w, scales, None, Some(biases), None, None, None),
                    MatmulB::ScaleZeroPointDequant {
                        b: w,
                        scales,
                        zero_points,
                        ..
                    } => (w, scales, Some(zero_points), None, None, None, None),
                    MatmulB::ScaleSymmetricDequant {
                        b: w,
                        scales,
                        ..
                    } => (w, scales, None, None, None, None, None),
                    MatmulB::LloydMaxDequant {
                        b: w,
                        scales,
                        codebook,
                        bias_indices,
                        bias_codebook,
                        ..
                    } => (w, scales, None, None, Some(codebook), Some(bias_indices), Some(bias_codebook)),
                    MatmulB::FullPrecision {
                        ..
                    } => unreachable!(),
                };
                pipeline.encode(
                    weights,
                    Some(scales),
                    zero_points,
                    biases,
                    codebook,
                    bias_indices,
                    bias_codebook,
                    (a, a_offset),
                    &mut *d,
                    output_bias,
                    rht_factors,
                    k,
                    n,
                    m,
                    ab_scale,
                    group_count_x,
                    encoder,
                );
            },
        }

        Ok(())
    }
}
