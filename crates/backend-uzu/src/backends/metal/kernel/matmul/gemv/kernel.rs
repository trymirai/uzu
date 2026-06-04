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
const QMV_PACKS_PER_THREAD: u32 = 2;
const DEFAULT_QMV_NUM_SIMDGROUPS: u32 = 8;
const DEFAULT_QMV_K_SPLIT: u32 = 1;
const QMV_NUM_SIMDGROUPS_ENV: &str = "UZU_QMV_NUM_SIMDGROUPS";

const DEFAULT_GEMV_MAX_BATCH: u32 = 8;
static GEMV_MAX_BATCH: OnceLock<u32> = OnceLock::new();
static QMV_NUM_SIMDGROUPS: OnceLock<u32> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct QmvSchedule {
    num_simdgroups: u32,
}

impl QmvSchedule {
    const DEFAULT: QmvSchedule = QmvSchedule {
        num_simdgroups: DEFAULT_QMV_NUM_SIMDGROUPS,
    };
}

fn max_gemv_batch_threshold() -> u32 {
    *GEMV_MAX_BATCH.get_or_init(|| {
        std::env::var("UZU_GEMV_MAX_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_GEMV_MAX_BATCH)
    })
}

fn env_choice(
    name: &str,
    default: u32,
    allowed: &[u32],
) -> u32 {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).filter(|v| allowed.contains(v)).unwrap_or(default)
}

fn qmv_num_simdgroups() -> u32 {
    *QMV_NUM_SIMDGROUPS.get_or_init(|| env_choice(QMV_NUM_SIMDGROUPS_ENV, DEFAULT_QMV_NUM_SIMDGROUPS, &[2, 4, 8]))
}

fn has_explicit_qmv_schedule_env() -> bool {
    std::env::var_os(QMV_NUM_SIMDGROUPS_ENV).is_some()
}

fn supports_qmv_schedule_tuning(
    b_prologue: GemmBPrologueKind,
    bits: u32,
    group_size: u32,
) -> bool {
    b_prologue == GemmBPrologueKind::ScaleBiasDequant && bits == 4 && group_size == 32
}

fn qmv_schedule(
    _bits: u32,
    _group_size: u32,
    _b_prologue: GemmBPrologueKind,
) -> QmvSchedule {
    QmvSchedule {
        num_simdgroups: qmv_num_simdgroups(),
    }
}

fn qmv_preheat_schedules(
    b_prologue: GemmBPrologueKind,
    bits: u32,
    group_size: u32,
) -> Vec<QmvSchedule> {
    let env_schedule = qmv_schedule(bits, group_size, b_prologue);
    if has_explicit_qmv_schedule_env() {
        return vec![env_schedule];
    }
    vec![env_schedule]
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
        let has_rht = args.d_transform.rht_factors.is_some();
        let b_prologue = args.b.b_prologue();
        let bits = args.b.bits_per_b().unwrap_or(0);
        let group_size = args.b.group_size().unwrap_or(0);
        let tune_quant_schedule = is_quant && !has_rht && supports_qmv_schedule_tuning(b_prologue, bits, group_size);
        let qmv_schedule = if tune_quant_schedule {
            qmv_schedule(bits, group_size, b_prologue)
        } else {
            QmvSchedule::DEFAULT
        };

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

        let block_size = match bits {
            4 => 512,
            8 => 256,
            _ => FP_BLOCK,
        };
        let input_aligned = args.k.is_multiple_of(block_size);
        let k_split = if is_quant || has_rht {
            DEFAULT_QMV_K_SPLIT
        } else {
            fp_k_split(args.n, args.k, input_aligned)
        };
        Some(GemvSpecialization {
            b_prologue,
            group_size,
            bits,
            output_transform: args.d_transform.mask(),
            input_aligned,
            k_split,
            num_simdgroups: qmv_schedule.num_simdgroups,
        })
    }

    fn quant_combo_specs(combo: MatmulQuantCombo) -> Vec<GemvSpecialization> {
        let bits = DataType::from(combo.mode).size_in_bits() as u32;
        let group_size = combo.group_size;
        let b_prologue = match combo.method {
            QuantizationMethod::ScaleBias => GemmBPrologueKind::ScaleBiasDequant,
            QuantizationMethod::ScaleZeroPoint => GemmBPrologueKind::ScaleZeroPointDequant,
            QuantizationMethod::ScaleSymmetric => GemmBPrologueKind::ScaleSymmetricDequant,
        };
        let mut out = Vec::new();
        for output_transform in [
            GemmDTransform::empty(),
            GemmDTransform::BIAS,
            GemmDTransform::RHT,
            GemmDTransform::BIAS | GemmDTransform::RHT,
        ] {
            for input_aligned in [true, false] {
                let schedules = if !output_transform.contains(GemmDTransform::RHT)
                    && supports_qmv_schedule_tuning(b_prologue, bits, group_size)
                {
                    qmv_preheat_schedules(b_prologue, bits, group_size)
                } else {
                    vec![QmvSchedule::DEFAULT]
                };
                for schedule in schedules {
                    out.push(GemvSpecialization {
                        b_prologue,
                        group_size,
                        bits,
                        output_transform,
                        input_aligned,
                        k_split: DEFAULT_QMV_K_SPLIT,
                        num_simdgroups: schedule.num_simdgroups,
                    });
                }
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
                    QMV_RESULTS_PER_SIMDGROUP,
                    QMV_PACKS_PER_THREAD,
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
            }) => {
                let (weights, scales, zero_points, biases) = match quant_b {
                    MatmulB::ScaleBiasDequant {
                        b: w,
                        scales,
                        biases,
                        ..
                    } => (w, scales, None, Some(biases)),
                    MatmulB::ScaleZeroPointDequant {
                        b: w,
                        scales,
                        zero_points,
                        ..
                    } => (w, scales, Some(zero_points), None),
                    MatmulB::ScaleSymmetricDequant {
                        b: w,
                        scales,
                        ..
                    } => (w, scales, None, None),
                    MatmulB::FullPrecision {
                        ..
                    } => unreachable!(),
                };
                pipeline.encode(
                    weights,
                    Some(scales),
                    zero_points,
                    biases,
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
